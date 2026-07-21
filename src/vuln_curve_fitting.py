import numpy as np
import pandas as pd
from scipy.stats import t
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

def calc_ls_ratio(df, y_mean):
    """
    Calculate loss ratio using detrended mean yield as the reference.

    Simulated yield values above the detrended mean are clipped to the
    detrended mean before calculating the loss ratio.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a 'SimulatedYield' column.
    y_mean : float
        Detrended mean yield used as the reference yield.

    Returns
    -------
    pandas.DataFrame
        DataFrame with an added 'ls_ratio' column.
    """
    df = df.copy()

    if y_mean <= 0:
        raise ValueError("y_mean must be greater than zero.")

    df["yield_sim_t_ha"] = df["yield_sim_t_ha"].clip(upper=y_mean)
    df["ls_ratio"] = 1 - df["yield_sim_t_ha"] / y_mean

    return df

def weighted_stats(g, alpha=0.05):
    w = g["harvest_area_ha"]
    x = g["ls_ratio"]

    mean = np.average(x, weights=w)

    # weighted population standard deviation
    wsd = np.sqrt(np.average((x - mean) ** 2, weights=w))

    n = len(g)
    t_star = t.ppf(1 - alpha/2, n-1)
    se = wsd / np.sqrt(n)
    ci_low = mean - t_star * se
    ci_high = mean + t_star * se

    return pd.Series({
        "weighted_mean_lr": mean,
        "weighted_sd_lr": wsd,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n": len(g),
        "weight_sum": w.sum()
    })

def binned_statistics(df, bins=10, alpha=0.05):
    """
    Compute binned means and confidence intervals for y_col based on x_col.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing x_col and y_col.
    num_bins : int, default=10
        Number of equal-width bins.
    alpha : float, default=0.05
        Significance level for the confidence interval.

    Returns
    -------
    df
        df containing:
        - weighted_mean_lr : weighted mean of loss ratio
        - weighted_sd_lr : weighted sd of loss ratio
        - se : 
        - ci_low : 
        - ci_high : 
        - n : sample size
        - weight_sum : 
    """
    df = df.copy()

    # customize bins here
    df["spei_bin"] = pd.cut(df["spei"], bins=bins, right=False)

    result = (
        df.groupby("spei_bin", observed=True)
          .apply(weighted_stats, alpha=alpha, include_groups=False)
          .reset_index()
    )

    return result

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

def fit_logistic_ranged(
    binned_x,
    binned_y,
    p0=None,
    prepend_origin=True,
):
    """
    Fit a logistic curve with a fixed lower asymptote of zero.

    Parameters
    ----------
    binned_x : array-like
        Binned x values for fitting.
    binned_y : array-like
        Binned y values for fitting.
    p0 : array-like, optional
        Initial guess for the parameters (U, k, x0).
    prepend_origin : bool, default=True
        Whether to prepend the point (0, 0) before fitting.

    Returns
    -------
    tuple
        (popt, perr, x_fit, y_fit, y_upper, y_lower)
    """
    if p0 is None:
        p0 = np.array([1.0, -1.0, -1.4])

    lower_bounds = [0.1, -np.inf, -np.inf]
    upper_bounds = [1.0, 0.0, 0.0]

    if prepend_origin:
        fit_x = np.insert(binned_x, 0, 0.0)
        fit_y = np.insert(binned_y, 0, 0.0)
    else:
        fit_x = binned_x
        fit_y = binned_y

    popt, pcov = curve_fit(
        logistic_ranged,
        fit_x,
        fit_y,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
    )

    perr = np.sqrt(np.diag(pcov))

    y_pred = logistic_ranged(binned_x, *popt)
    rmse = np.sqrt(mean_squared_error(binned_y, y_pred))

    x_fit = np.linspace(-3.0, 0.0, 100)
    y_fit = logistic_ranged(x_fit, *popt)
    y_upper = logistic_ranged(x_fit, *(popt + perr))
    y_lower = logistic_ranged(x_fit, *(popt - perr))

    return popt, perr, rmse, x_fit, y_fit, y_upper, y_lower