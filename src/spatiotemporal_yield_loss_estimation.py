import numpy as np

def get_trend_polynomial(values, degree=2):
    """
    Fit a polynomial trend to input values and return the trend values
    without the constant offset component.

    Args:
        values (array-like): Input data points.
        degree (int): Degree of the polynomial fit.

    Returns:
        np.ndarray: Polynomial trend values with the constant term removed.
    """
    x_values = np.arange(len(values))

    # Fit polynomial: y = a_n*x^n + ... + a_1*x + a_0
    coefficients = np.polyfit(x_values, values, degree)

    # Remove constant term
    coefficients[-1] = 0

    # Create polynomial function
    polynomial = np.poly1d(coefficients)

    # Return polynomial trend values without constant offset
    return polynomial(x_values)

def logistic_ranged(x, upper, growth_rate, midpoint):
    """
    Compute a logistic function with a fixed lower asymptote of 0.

    Parameters
    ----------
    x : float or numpy.ndarray
        Input value(s).
    upper : float
        Upper asymptote.
    growth_rate : float
        Growth rate of the logistic curve.
    midpoint : float
        Inflection point (the x-value at which the function reaches
        half of the upper asymptote).

    Returns
    -------
    float or numpy.ndarray
        Logistic function evaluated at `x`.
    """
    lower = 0.0
    return lower + (upper - lower) / (
        1 + np.exp(-growth_rate * (x - midpoint))
    )

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

def construct_lec_curve(loss_series):
    """Construct a loss exceedance curve (LEC).

    Parameters
    ----------
    loss_series : array-like
        Series of loss values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Return periods and corresponding losses sorted in descending order.
    """
    sorted_losses = np.sort(loss_series)[::-1]
    n_losses = len(sorted_losses)

    ranks = np.arange(1, n_losses + 1)
    exceedance_probability = ranks / (n_losses + 1)
    return_period = 1 / exceedance_probability

    return return_period, sorted_losses