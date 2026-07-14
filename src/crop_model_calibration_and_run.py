from pathlib import Path
import numpy as np
import pandas as pd
import copy
from aquacrop import AquaCropModel, Crop, InitialWaterContent, Soil


def n_for_coverage(values, threshold=0.9):
    """
    Return the minimum number of top elements required to
    reach a given cumulative coverage threshold.

    Parameters
    ----------
    values : array-like
        Numeric values (e.g., crop harvested area per grid cell).
    threshold : float
        Fraction of total sum to cover (default = 0.9).

    Returns
    -------
    int
        Number of top elements needed to reach the threshold.
    """
    values_sorted = np.sort(values.to_numpy())[::-1]  # descending order
    cumulative_sum = np.cumsum(values_sorted)
    total_sum = cumulative_sum[-1]

    # First index where cumulative coverage exceeds threshold
    cutoff_index = np.searchsorted(cumulative_sum, threshold * total_sum)

    return cutoff_index + 1  # convert from 0-based index to count


def get_gid_ha_by_coverage(df, threshold=0.9):
    """
    Select GIDs that together account for a given fraction of total crop
    area.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'GID' and crop harvest area column.
    threshold : float
        Fraction of total crop area to retain (default: 0.9).

    Returns
    -------
    selected_df
        GID and harest area for grids contributing to the top coverage
        threshold.
    """
    required_columns = ["GID", "harvest_area_ha"]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]
    
    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}. "
            f"Available columns: {list(df.columns)}"
        )
    
    # Keep only relevant columns
    data = df[required_columns]

    # Number of top grid cells required to reach coverage threshold
    top_n = n_for_coverage(data["harvest_area_ha"], threshold)

    # Number of non-zero grid cells (for reporting)
    nonzero_grids = (data["harvest_area_ha"] > 0).sum()

    print(
        f"{top_n} out of {nonzero_grids} grids selected for calibration, "
        f"accounting for {threshold:.0%} of area."
    )

    # Identify top contributing grid indices
    top_indices = data["harvest_area_ha"].nlargest(top_n).index

    # Extract corresponding GIDs safely using label-based indexing
    selected_df = data.loc[top_indices, ["GID", "harvest_area_ha"]]

    return selected_df


def format_aquacrop_weather(weather_file_path: Path) -> pd.DataFrame:
    """
    Read and format AquaCrop weather data.

    Parameters
    ----------
    weather_file_path : Path
        Path to the tab-separated weather file.

    Returns
    -------
    pd.DataFrame
        Cleaned weather dataframe with datetime index and clipped ET0
        values.
    """

    # Load dataset with 7 expected columns:
    # - Day
    # - Month
    # - Year
    # - Tmin
    # - Tmax
    # - Precipitation
    # - ET0
    weather_df = pd.read_csv(weather_file_path, header=0, sep="\t")

    if weather_df.shape[1] != 7:
        raise ValueError(
            f"Expected 7 columns in {weather_file_path}, "
            f"got {weather_df.shape[1]}"
        )

    # Assign explicit column names for stability
    weather_df.columns = [
        "Day", "Month", "Year",
        "MinTemp", "MaxTemp",
        "Precipitation", "ReferenceET"
    ]

    # Create datetime column safely
    weather_df["Date"] = pd.to_datetime(
        weather_df[["Year", "Month", "Day"]],
        errors="coerce"
    )

    # Remove redundant date components
    weather_df.drop(columns=["Day", "Month", "Year"], inplace=True)

    # Prevent division issues in downstream models
    weather_df["ReferenceET"] = weather_df["ReferenceET"].clip(lower=0.1)

    return weather_df


def get_weather_dfs(gid_list, weather_folder_path: Path):
    """
    Load and filter AquaCrop weather datasets for multiple grid IDs.

    Parameters
    ----------
    gid_list : list
        List of grid IDs.
    weather_folder_path : Path
        Path to the tab-separated weather folder.

    Returns
    -------
    tuple[list, list]
        - Valid GIDs with acceptable data quality
        - Corresponding list of cleaned weather DataFrames
    """

    print(f"Total grids provided: {len(gid_list)}")

    valid_gids = []
    weather_dfs = []

    # Allow up to 1% missing values per column
    missing_threshold = 0.01

    for gid in gid_list:
        file_path = weather_folder_path / f"{gid}_climate.txt"

        # Skip missing files
        if not file_path.exists():
            continue

        # Load and format data
        df = format_aquacrop_weather(file_path)

        # Fill missing values (time-series safe)
        df = df.ffill().bfill()

        # Quality check: fraction of missing values per column
        if df.isna().mean().max() <= missing_threshold:
            valid_gids.append(gid)
            weather_dfs.append(df)

    print(f"Valid grids after data coverage filtering: {len(valid_gids)}")

    return valid_gids, weather_dfs


def simulate_grid_yield(
    crop_parameters,
    crop_name,
    weather_data,
    planting_date="05/01",
    sim_start_date="1991/01/01",
    sim_end_date="2020/12/31",
    soil_type="ClayLoam",
):
    """
    Run an AquaCrop simulation and return the simulated grain yield.

    Parameters
    ----------
    crop_parameters : dict
        Crop parameters passed to the Crop constructor.
    crop_name : str
        AquaCrop crop name (e.g. "PaddyRiceGDD").
    weather_data : pd.DataFrame
        Daily weather data for the simulation period.
    planting_date : str, optional
        Planting date in "MM/DD" format.
    sim_start_date : str, optional
        Simulation start date.
    sim_end_date : str, optional
        Simulation end date.
    soil_type : str, optional
        Soil type defined in AquaCrop.

    Returns
    -------
    pd.Series
        Simulated grain yield (tonne/ha).
    """

    # Define soil properties
    soil = Soil(soil_type=soil_type)

    # Create crop object
    crop = Crop(
        crop_name,
        planting_date=planting_date,
        **crop_parameters,
    )

    # Initialize soil water content at field capacity
    initial_water = InitialWaterContent(value=["FC"])

    # Create and run AquaCrop model
    model = AquaCropModel(
        sim_start_time=sim_start_date,
        sim_end_time=sim_end_date,
        weather_df=weather_data,
        soil=soil,
        crop=crop,
        initial_water_content=initial_water,
    )

    model.run_model(till_termination=True)

    return model._outputs.final_stats["Dry yield (tonne/ha)"]


def aggregate_yearly_yield(df, yield_threshold=0.01):
    df = df.copy()

    # Filter out low-yield records (i.e., crop failure pixel-year).
    df = df.loc[df["yield_sim_t_ha"] >= yield_threshold]

    # Calculate production (t) for each record.
    df["production_t"] = df["yield_sim_t_ha"] * df["harvest_area_ha"]

    # Aggregate total production and harvested area by year.
    yearly = (
        df.groupby("year", as_index=False)
        .agg(
            production_t=("production_t", "sum"),
            harvest_area_ha=("harvest_area_ha", "sum"),
        )
    )

    # Compute area-weighted average yield (t/ha).
    yearly["yield_sim_t_ha"] = (
        yearly["production_t"] / yearly["harvest_area_ha"]
    )

    return yearly[["year", "yield_sim_t_ha"]]


def weighted_mean_squared_error(y_true, y_pred, sample_weights=None):
    """Compute the weighted mean squared error (MSE).

    Args:
        y_true (array-like): Ground-truth target values.
        y_pred (array-like): Predicted target values.
        sample_weights (array-like): Non-negative weight for each
        sample.

    Returns:
        float: Weighted mean squared error.

    Raises:
        ValueError: If the input arrays have different lengths, the
        weights contain negative values, or the sum of weights is zero.
    """
    if sample_weights is None:
        sample_weights = np.ones(len(y_true))
    # Convert inputs to NumPy arrays.
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sample_weights = np.asarray(sample_weights, dtype=float)

    # Validate input dimensions.
    if not (
        len(y_true) == len(y_pred) == len(sample_weights)
    ):
        raise ValueError(
            "y_true, y_pred, and sample_weights must have the same length."
        )

    if np.any(sample_weights < 0):
        raise ValueError("Sample weights must be non-negative.")

    total_weight = np.sum(sample_weights)
    if total_weight == 0:
        raise ValueError(
            "The sum of sample weights must be greater than zero."
        )

    # Compute the squared prediction errors.
    squared_errors = np.square(y_true - y_pred)

    # Compute the weighted average of the squared errors.
    weighted_mse = np.sum(
        squared_errors * sample_weights
    ) / total_weight

    return weighted_mse


def make_asymmetric_parameter_space(
    param_values,
    lower_factor,
    upper_factor,
):
    parameter_limits = {
        "CCx": (0.0, 0.99),
        "WPy": (0.0, 100.0),
        "p_up1": (0.0, 0.2),
        "p_up2": (0.3, 0.51),
        "p_up3": (0.52, 0.7),
    }

    parameter_space = {}

    for k, v in param_values.items():
        if k == "CDC":
            lower = (2 - upper_factor) * v
            upper = (2 - lower_factor) * v
        else:
            lower = v * lower_factor
            upper = v * upper_factor

        low_limit, high_limit = parameter_limits.get(
            k, (-float("inf"), float("inf"))
        )

        parameter_space[k] = (
            max(lower, low_limit),
            min(upper, high_limit),
        )

    return parameter_space


def make_objective(
    gid_harvest_area_df,
    weather_df_list,
    observed_yield_df,
    parameter_space,
    crop_name,
    planting_date="05/01",
    user_harvest_year=None,
    sim_start_time="1991/01/01",
    sim_end_time="2020/12/31",
    soil_type="ClayLoam",
    yield_threshold=0.01,
    sample_weights=None,
):
    """Create an Optuna objective function for calibrating crop
    parameters.
    """
    def objective(trial):
        # Create crop object
        crop = Crop(
            crop_name,
            planting_date=planting_date,
        )

        # Update the parameters being optimized.
        for parameter, (lower, upper) in parameter_space.items():
            setattr(
                crop,
                parameter,
                trial.suggest_float(parameter, lower, upper),
            )

        simulation_results = []

        for gid, weather_df, harvest_area in zip(
            gid_harvest_area_df["GID"],
            weather_df_list,
            gid_harvest_area_df["harvest_area_ha"],
        ):
            model = AquaCropModel(
                sim_start_time=sim_start_time,
                sim_end_time=sim_end_time,
                weather_df=weather_df,
                soil=Soil(soil_type=soil_type),
                crop=crop,
                initial_water_content=InitialWaterContent(value=["FC"]),
            )

            model.run_model(till_termination=True)

            final_stats = model._outputs.final_stats
            harvest_year = (
                final_stats["Harvest Date (YYYY/MM/DD)"].dt.year
                if user_harvest_year is None
                else user_harvest_year
            )
            y_simulated = final_stats["Dry yield (tonne/ha)"]

            simulation_results.append(
                pd.DataFrame(
                    {
                        "GID": gid,
                        "year": harvest_year,
                        "yield_sim_t_ha": y_simulated[:len(harvest_year)],
                        "harvest_area_ha": harvest_area,
                    }
                )
            )

        simulated_yield_df = aggregate_yearly_yield(
            pd.concat(simulation_results, ignore_index=True),
            yield_threshold
        )

        comparison_df = simulated_yield_df.merge(
            observed_yield_df,
            on="year",
            how="left",
        )

        return weighted_mean_squared_error(
            y_true=comparison_df["yield_obs_t_ha"],
            y_pred=comparison_df["yield_sim_t_ha"],
            sample_weights=sample_weights,
        )

    return objective


def simulate_country_yield(
    gid_harvest_area_df,
    weather_df_list,
    observed_yield_df,
    best_crop_parameters,
    crop_name,
    planting_date="05/01",
    user_harvest_year=None,
    sim_start_time="1991/01/01",
    sim_end_time="2020/12/31",
    soil_type="ClayLoam",
    yield_threshold=0.01,
):
    """
    Run an AquaCrop simulation and return the simulated grain yield.

    Returns
    -------
    pd.Series
        Simulated grain yield (tonne/ha).
    """
    
    # Create crop object
    crop = Crop(
        crop_name,
        planting_date=planting_date,
        **best_crop_parameters,
    )

    simulation_results = []

    for gid, weather_df, harvest_area in zip(
        gid_harvest_area_df["GID"],
        weather_df_list,
        gid_harvest_area_df["harvest_area_ha"],
    ):
        model = AquaCropModel(
            sim_start_time=sim_start_time,
            sim_end_time=sim_end_time,
            weather_df=weather_df,
            soil=Soil(soil_type=soil_type),
            crop=crop,
            initial_water_content=InitialWaterContent(value=["FC"]),
        )

        model.run_model(till_termination=True)

        final_stats = model._outputs.final_stats
        harvest_year = (
            final_stats["Harvest Date (YYYY/MM/DD)"].dt.year
            if user_harvest_year is None
            else user_harvest_year
        )
        harvest_date = final_stats["Harvest Date (YYYY/MM/DD)"]
        y_simulated = final_stats["Dry yield (tonne/ha)"]

        simulation_results.append(
            pd.DataFrame(
                {
                    "GID": gid,
                    "year": harvest_year,
                    "planting_date": planting_date,
                    "harvest_date": harvest_date[:len(harvest_year)],
                    "yield_sim_t_ha": y_simulated[:len(harvest_year)],
                    "harvest_area_ha": harvest_area,
                }
            )
        )

    simulation_results_df = pd.concat(simulation_results, ignore_index=True)
    
    simulated_yield_df = aggregate_yearly_yield(
        simulation_results_df,
        yield_threshold
    )
    
    comparison_df = simulated_yield_df.merge(
        observed_yield_df,
        on="year",
        how="left",
    )

    mse = weighted_mean_squared_error(
        comparison_df["yield_obs_t_ha"],
        comparison_df["yield_sim_t_ha"],
    )

    print(f"RMSE (t/ha): {np.sqrt(mse):.3f}")

    return simulation_results_df, comparison_df


def summarize_productive_grids(
    df: pd.DataFrame,
    yield_threshold: float = 0.1,
) -> pd.DataFrame:
    """
    Calculate, for each year:
      - Number of unique GIDs with yield > yield_threshold.
      - Total harvested area where yield > yield_threshold.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns:
        ["GID", "year", "yield_sim_t_ha", "harvest_area_ha"].
    yield_threshold : float, default=0.1
        Yield threshold (t/ha).

    Returns
    -------
    pd.DataFrame
        Columns:
        - year
        - n_gid
        - harvest_area_ha
    """
    df_filtered = df[df["yield_sim_t_ha"] >= yield_threshold]

    return (
        df_filtered.groupby("year")
        .agg(
            n_gid=("GID", "nunique"),
            harvest_area_ha=("harvest_area_ha", "sum"),
        )
        .reset_index()
    )