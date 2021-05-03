# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Data processing utility functions for curve fitting experiments
"""
# pylint: disable = invalid-name

from typing import List, Dict, Tuple, Optional
import numpy as np
from qiskit.exceptions import QiskitError


def filter_data(data: List[Dict[str, any]], **filters) -> List[Dict[str, any]]:
    """Return the list of filtered data

    Args:
        data: list of data dicts.
        filters: kwargs for filtering based on metadata
                 values.

    Returns:
        The list of filtered data. If no filters are provided this will be the
        input list.
    """
    if not filters:
        return data
    filtered_data = []
    for datum in data:
        include = True
        metadata = datum["metadata"]
        for key, val in filters.items():
            if key not in metadata or metadata[key] != val:
                include = False
                break
        if include:
            filtered_data.append(datum)
    return filtered_data

def mean_xy_data(
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    method: str = "sample",
) -> Tuple[np.ndarray]:
    r"""Wrapper function for multi_mean_xy_data
    in the case of one data series
    """
    series = np.zeros(xdata.size)
    x_means, y_means, y_sigmas, _ = multi_mean_xy_data(xdata, ydata, sigma, series, method)
    return x_means, y_means, y_sigmas

def multi_mean_xy_data(
    xdata: np.ndarray,
    ydata: np.ndarray,
    sigma: Optional[np.ndarray] = None,
    series: Optional[np.ndarray] = None,
    method: str = "sample",
) -> Tuple[np.ndarray]:
    r"""Return (x, y_mean, sigma, series) data.

    The mean is taken over all ydata values with the same xdata value using
    the specified method. For each x the mean :math:`\overline{y}` and variance
    :math:`\sigma^2` are computed as

    * ``"sample"`` (default) *Sample mean and variance*
      :math:`\overline{y} = \sum_{i=1}^N y_i / N`,
      :math:`\sigma^2 = \sum_{i=1}^N ((\overline{y} - y_i)^2) / N`
    * ``"iwv"`` *Inverse-weighted variance*
      :math:`\overline{y} = (\sum_{i=1}^N y_i / \sigma_i^2 ) \sigma^2`
      :math:`\sigma^2 = 1 / (\sum_{i=1}^N 1 / \sigma_i^2)`

    Args
        xdata: 1D array of xdata from curve_fit_data or
               multi_curve_fit_data
        ydata: array of ydata returned from curve_fit_data or
               multi_curve_fit_data
        series: 1D int array that specifies the data series
        sigma: Optional, array of standard deviations in ydata.
        method: The method to use for computing y means and
                standard deviations sigma (default: "sample").

    Returns:
        tuple: ``(x, y_mean, sigma)`` if ``return_raw==False``, where
               ``x`` is an arrays of unique x-values, ``y`` is an array of
               sample mean y-values, and ``sigma`` is an array of sample standard
               deviation of y values.

    Raises:
        QiskitError: if "ivw" method is used without providing a sigma.
    """
    if method == "iwv" and sigma is None:
        raise QiskitError(
            "The inverse-weighted variance method cannot be used with" " `sigma=None`"
        )
    if method not in ["sample", "iwv"]:
        raise QiskitError(f"Unsupported method {method}")
    x_keys = np.unique(np.column_stack((xdata, series)), axis=0)
    x_means = np.zeros(x_keys.shape[0])
    y_means = np.zeros(x_keys.shape[0])
    y_sigmas = np.zeros(x_keys.shape[0])
    series_mean = np.zeros(x_keys.shape[0])

    for i, (x_val, xseries_val) in enumerate(x_keys):
        x_means[i] = x_val
        series_mean[i] = xseries_val
        # Get positions of y to average
        idxs = np.where((xdata == x_val) & (series == xseries_val))
        ys = ydata[idxs]

        # Sample mean and variance method
        if method == "sample":
            # Compute sample mean and biased sample variance
            y_means[i] = np.mean(ys)
            y_sigmas[i] = np.mean((y_means[i] - ys) ** 2)

        # Inverse-weighted variance method
        if method == "iwv":
            # Compute the inverse-variance weighted y mean and variance
            weights = 1 / sigma[idxs] ** 2
            y_var = 1 / np.sum(weights)
            y_means[i] = y_var * np.sum(weights * ys)
            y_sigmas[i] = np.sqrt(y_var)

    return x_means, y_means, y_sigmas, series_mean


def level2_probability(data: Dict[str, any], outcome: str) -> Tuple[float]:
    """Return the outcome probability mean and variance.

    Args:
        data: A data dict containing count data.
        outcome: bitstring for desired outcome probability.

    Returns:
        tuple: (p_mean, p_var) of the probability mean and variance
               estimated from the counts.

    .. note::

        This assumes a binomial distribution where :math:`K` counts
        of the desired outcome from :math:`N` shots the
        mean probability is :math:`p = K / N` and the variance is
        :math:`\\sigma^2 = p (1-p) / N`.
    """
    counts = data["counts"]
    shots = sum(counts.values())
    p_mean = counts.get(outcome, 0.0) / shots
    p_var = p_mean * (1 - p_mean) / shots
    return p_mean, p_var
