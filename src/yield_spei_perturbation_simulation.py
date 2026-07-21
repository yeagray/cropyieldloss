import numpy as np
import pandas as pd
import xarray as xr

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

def get_valid_gids(selected_gids_df, grid_gdf, data_array):
    """
    Return GIDs whose nearest grid cell contains no more than 1% missing
    values.

    Parameters
    ----------
    selected_gids_df : pandas.DataFrame
        DataFrame containing the selected GIDs.
    grid_gdf : geopandas.GeoDataFrame
        GeoDataFrame containing GID centroid coordinates.
    data_array : xarray.DataArray
        Time series data with dimensions including latitude and
        longitude.

    Returns
    -------
    list
        List of valid GIDs.
    """
    # Attach centroid coordinates to the selected GIDs
    selected_points = selected_gids_df.merge(
        grid_gdf[["GID", "centroid_x", "centroid_y"]],
        on="GID",
        how="left",
    )
    
    valid_gids = []

    # Check data availability at each grid centroid
    for _, row in selected_points.iterrows():
        ts = data_array.sel(
            lon=row["centroid_x"],
            lat=row["centroid_y"],
            method="nearest"
        )
    
        missing_fraction = np.isnan(ts).mean().item()
    
        if missing_fraction <= 0.01:
            valid_gids.append(row["GID"])
    gid_harvest_area_valid_df = selected_points[
        selected_points["GID"].isin(valid_gids)
        ]
    return gid_harvest_area_valid_df[
        [
            "GID",
            "centroid_x",
            "centroid_y",
            "harvest_area_ha",
        ]
    ]

def add_year_month(
    df: pd.DataFrame,
    sim_start_time: str,
    sim_end_time: str,
) -> pd.DataFrame:
    """Expand each record into one row per month-end within the simulation period."""

    simulation_start = pd.to_datetime(sim_start_time, format="%Y/%m/%d")
    simulation_end = pd.to_datetime(sim_end_time, format="%Y/%m/%d")

    result = df.copy()

    month_end_dates = pd.date_range(
        start=simulation_start,
        end=simulation_end,
        freq="ME",
    )

    result["harvest_date"] = [month_end_dates] * len(result)

    result = result.explode("harvest_date", ignore_index=True)
    result["year"] = result["harvest_date"].dt.year

    return result[
        [
            "GID",
            "year",
            "harvest_date",
            "centroid_x",
            "centroid_y",
            "harvest_area_ha",
        ]
    ]

def add_spei(dataframe: pd.DataFrame, spei_data: xr.DataArray) -> pd.DataFrame:
    """
    Add SPEI values to a DataFrame based on harvest date and centroid
    coordinates.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Input DataFrame containing "harvest_date", "centroid_x", and
        "centroid_y" columns.
    spei_data : xr.DataArray
        SPEI data with "time", "lon", and "lat" dimensions.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added "spei" column.
    """
    dataframe = dataframe.copy()

    dataframe["spei"] = spei_data.sel(
        time=xr.DataArray(
            dataframe["harvest_date"].values,
            dims="points",
        ),
        lon=xr.DataArray(
            dataframe["centroid_x"].values,
            dims="points",
        ),
        lat=xr.DataArray(
            dataframe["centroid_y"].values,
            dims="points",
        ),
        method="nearest",
    ).values

    return dataframe

def calculate_mean_perturbation_by_spei(
    input_df: pd.DataFrame,
    beta: float,
    n_trials: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate mean SPEI-based yield perturbation factors.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame containing a "spei" column.

    beta : float
        SPEI-induced yield reduction sensitivity factor.

    n_trials : int, default=30
        Number of random perturbation simulations.

    Returns
    -------
    disturbance_df : pd.DataFrame
        Individual perturbation values for each trial.

    output_df : pd.DataFrame
        Input dataframe with added "mean_perturbation" column.
    """

    output_df = input_df.copy()

    n_rows = len(output_df)
    np.random.seed(42)

    perturbation_results = np.zeros((n_rows, n_trials))

    # SPEI severity classes
    mask_1 = (output_df["spei"] >= -1) & (output_df["spei"] < 0)
    mask_2 = (output_df["spei"] >= -1.5) & (output_df["spei"] < -1)
    mask_3 = (output_df["spei"] >= -2) & (output_df["spei"] < -1.5)
    mask_4 = output_df["spei"] < -2

    for trial in range(n_trials):
        perturbation_results[mask_1, trial] = np.random.uniform(
            -beta,
            0,
            mask_1.sum(),
        )

        perturbation_results[mask_2, trial] = np.random.uniform(
            -1.5 * beta,
            -beta,
            mask_2.sum(),
        )

        perturbation_results[mask_3, trial] = np.random.uniform(
            -2 * beta,
            -1.5 * beta,
            mask_3.sum(),
        )

        perturbation_results[mask_4, trial] = np.random.uniform(
            -1,
            -2 * beta,
            mask_4.sum(),
        )

    perturbation_df = pd.DataFrame(
        perturbation_results,
        columns=[f"trial_{i}" for i in range(n_trials)],
    )

    output_df["mean_perturbation"] = perturbation_df.mean(axis=1)

    return perturbation_df, output_df

def aggregate_yearly_yield(df, yield_threshold):
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

def scale_perturb_yield(
    input_df: pd.DataFrame,
    observed_yield_df: pd.DataFrame,
    yield_threshold: float,
) -> pd.DataFrame:
    """
    Scale disturbed yield to match observed yield.

    Parameters
    ----------
    input_df : pd.DataFrame
        DataFrame containing:
        - year
        - mean_perturbation
        - harvest_area_ha

    observed_yield_df : pd.DataFrame
        Observed yield dataframe containing "year" and "yield_obs_t_ha".

    Returns
    -------
    pd.DataFrame
        DataFrame with disturbed and scaled simulated yield.
    """

    output_df = input_df.copy()

    output_df = output_df.merge(observed_yield_df, on="year", how="left")

    if (output_df["yield_obs_t_ha"] < 0).any():
        print("Negative observed yield exists")
    
    # Apply disturbance
    output_df["yield_sim_t_ha"] = (
        output_df["yield_obs_t_ha"]
        * (1 + output_df["mean_perturbation"])
    )

    if (output_df["yield_sim_t_ha"] < 0).any():
        print("Negative simulated yield exists")

    simulated_yield_df = aggregate_yearly_yield(output_df, yield_threshold)

    comparison_df = simulated_yield_df.merge(
        observed_yield_df,
        on="year",
        how="left",
    )

    # Scale simulated yield to observed yield
    comparison_df["scale_factor"] = (
        comparison_df["yield_obs_t_ha"]
        / comparison_df["yield_sim_t_ha"]
    )

    if comparison_df["year"].nunique() < output_df["year"].nunique():
        print("Some year does not have any valid grids")
        print("Grids in these years are assigned with zero yield")
    
    output_df = output_df.merge(
        comparison_df[["year", "scale_factor"]],
        on="year",
        how="left",
    )
    output_df["scale_factor"] = output_df["scale_factor"].fillna(0)

    output_df["yield_scaled_t_ha"] = (
        output_df["scale_factor"]
        * output_df["yield_sim_t_ha"]
    )

    return output_df

def simulate_monthly_yield(
    gid_harvest_area_selected_df,
    gdf,
    da,
    observed_yield_df,
    beta=0.3,
    sim_start_time="2016/01/01",
    sim_end_time="2020/12/31",
    n_trials=30,
    yield_threshold=0,
):
    gid_harvest_area_valid_df = get_valid_gids(
        gid_harvest_area_selected_df,
        gdf,
        da,
    )

    harvest_df = add_year_month(
        gid_harvest_area_valid_df,
        sim_start_time,
        sim_end_time,
    )

    harvest_spei_df = add_spei(harvest_df, da)

    _, simulation_df = calculate_mean_perturbation_by_spei(
        harvest_spei_df,
        beta,
        n_trials,
    )

    simulation_results_df = scale_perturb_yield(
        simulation_df,
        observed_yield_df,
        yield_threshold,
    )
    return simulation_results_df

def add_plant_harvest_dates(
    df: pd.DataFrame,
    sim_start_time: str,
    sim_end_time: str,
    planting_date: str,
    growing_months: int,
) -> pd.DataFrame:
    """
    Add planting and harvest dates for all possible harvest cycles
    within the simulation period.
    """

    start_date = pd.to_datetime(sim_start_time, format="%Y/%m/%d")
    end_date = pd.to_datetime(sim_end_time, format="%Y/%m/%d")

    planting_years = range(start_date.year, end_date.year + 1)

    harvest_schedule = pd.DataFrame(
        {"planting_year": planting_years}
    )

    harvest_schedule["planting_date"] = pd.to_datetime(
        harvest_schedule["planting_year"].astype(str) + "/" + planting_date,
        format="%Y/%m/%d",
    )

    harvest_schedule["harvest_date"] = (
        harvest_schedule["planting_date"]
        + pd.DateOffset(months=growing_months)
    )

    # Use "year" for harvest year
    harvest_schedule["year"] = harvest_schedule["harvest_date"].dt.year

    harvest_schedule = harvest_schedule[
        harvest_schedule["harvest_date"].between(start_date, end_date)
    ].reset_index(drop=True)

    result = df.merge(
        harvest_schedule[
            ["year", "planting_date", "harvest_date"]
        ],
        how="cross",
    )

    return result

def add_mean_spei(df, spei_data_array):
    """
    Add mean SPEI values between planting_date and harvest_date for each
    row in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing centroid_x, centroid_y, planting_date, and
        harvest_date columns.
    spei_data_array : xarray.DataArray
        SPEI DataArray with dimensions lon, lat, and time.

    Returns
    -------
    pandas.DataFrame
        DataFrame with a new 'spei' column.
    """
    df = df.copy()

    df["spei"] = [
        spei_data_array.sel(
            lon=row["centroid_x"],
            lat=row["centroid_y"],
            time=slice(row["planting_date"], row["harvest_date"]),
        )
        .mean(dim="time")
        .item()
        for _, row in df.iterrows()
    ]

    return df

def simulate_yearly_yield(
    gid_harvest_area_selected_df,
    gdf,
    da,
    observed_yield_df,
    beta,
    planting_date,
    growing_months,
    sim_start_time="1991/01/01",
    sim_end_time="2020/12/31",
    n_trials=30,
    yield_threshold=0,
):
    gid_harvest_area_valid_df = get_valid_gids(
        gid_harvest_area_selected_df,
        gdf,
        da,
    )

    harvest_df = add_plant_harvest_dates(
        gid_harvest_area_valid_df,
        sim_start_time,
        sim_end_time,
        planting_date,
        growing_months,
    )

    harvest_spei_df = add_mean_spei(harvest_df, da)

    _, simulation_df = calculate_mean_perturbation_by_spei(
        harvest_spei_df,
        beta,
        n_trials,
    )

    simulation_results_df = scale_perturb_yield(
        simulation_df,
        observed_yield_df,
        yield_threshold,
    )
    return simulation_results_df