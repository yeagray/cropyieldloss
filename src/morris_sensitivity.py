from pathlib import Path
import numpy as np
from SALib.sample.morris import sample
from SALib.analyze.morris import analyze
from crop_model_calibration_and_run import get_weather_dfs, simulate_grid_yield


def create_sampling_problem(
    crop_params,
    lower_factor=0.7,
    upper_factor=1.1,
    parameter_limits=None,
):
    """
    Create a SALib sampling problem from crop parameters.

    Parameters
    ----------
    crop_params : dict
        Nominal crop parameter values.
    lower_factor : float, optional
        Lower sampling factor relative to the nominal value.
    upper_factor : float, optional
        Upper sampling factor relative to the nominal value.
    parameter_limits : dict, optional
        Lower and upper limits for selected parameters.
        Example:
            {
                "CCx": (0.0, 0.99),
                "WPy": (0.0, 100.0),
            }

    Returns
    -------
    dict
        SALib problem definition.
    """

    if parameter_limits is None:
        parameter_limits = {
            "CCx": (0.0, 0.99),
            "WPy": (0.0, 100.0),
            "p_up1": (0, 0.2),
            "p_up2": (0.3, 0.51),
            "p_up3": (0.52,0.7),
        }

    bounds = []

    for name, value in crop_params.items():
        lower = lower_factor * value
        upper = upper_factor * value
        if name == "CDC":
            lower = (2-upper_factor) * value
            upper = (2-lower_factor) * value

        # Apply parameter-specific limits
        if name in parameter_limits:
            min_value, max_value = parameter_limits[name]
            lower = max(lower, min_value)
            upper = min(upper, max_value)

        bounds.append([lower, upper])

    return {
        "num_vars": len(crop_params),
        "names": list(crop_params.keys()),
        "bounds": bounds,
    }


def compute_morris_indices(problem, samples, outputs):
    """
    Computes Morris sensitivity indices for each output dimension.
    Returns:
        list of dicts: one per output variable
    """
    return [
        analyze(
            problem,
            samples,
            outputs[:, output_idx],
            conf_level=0.95,
            print_to_console=False,
        )
        for output_idx in range(outputs.shape[1])
    ]


def run_morris_analysis(
    gid_list,
    weather_folder,
    crop_params,
    crop_name="PaddyRiceGDD",
    planting_date="05/01",
    sim_start_date="1991/01/01",
    sim_end_date="2020/12/31",
    soil_type="ClayLoam",
    n_outputs=30,
    lower_factor=0.8,
    upper_factor=1.1,
    n_trajectories=20,
    num_levels=6,
):
    """Run a complete Morris sensitivity analysis for a single grid.

    Returns
    -------
    dict
        {
            "problem": problem_definition,
            "samples": parameter_samples,
            "outputs": model_outputs,
            "indices": sensitivity_indices,
        }
    """
    # Load weather data
    _, weather_dfs = get_weather_dfs(gid_list, Path(weather_folder))
    weather_data = weather_dfs[0]

    # Define Morris problem
    problem_definition = create_sampling_problem(
        crop_params,
        lower_factor=lower_factor,
        upper_factor=upper_factor,
    )

    # Generate parameter samples
    parameter_samples = sample(
        problem_definition,
        N=n_trajectories,
        num_levels=num_levels,
        optimal_trajectories=None,
    )

    # Run model
    n_samples = len(parameter_samples)
    model_outputs = np.zeros((n_samples, n_outputs))

    drop_last_col = False
    for sample_idx, parameters in enumerate(parameter_samples):
        crop_parameters = dict(zip(crop_params, parameters))
        y_simulated = simulate_grid_yield(
            crop_parameters,
            crop_name,
            weather_data,
            planting_date=planting_date,
            soil_type=soil_type,
            sim_start_date=sim_start_date,
            sim_end_date=sim_end_date,
        )
        model_outputs[sample_idx, :len(y_simulated)] = y_simulated
        if len(y_simulated) < n_outputs:
            drop_last_col = True

    # Trim outputs
    if drop_last_col:
        model_outputs = model_outputs[:, :-1]
        print(f"model_outputs shape: {model_outputs.shape}")

    # Compute Morris indices
    sensitivity_indices = compute_morris_indices(
        problem_definition,
        parameter_samples,
        model_outputs,
    )

    return {
        "problem": problem_definition,
        "samples": parameter_samples,
        "outputs": model_outputs,
        "indices": sensitivity_indices,
    }