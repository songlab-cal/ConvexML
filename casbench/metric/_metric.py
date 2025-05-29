"""
Evaluation metrics.
"""
from typing import Callable, List

import numpy as np
import scipy
import scipy.special


def sr2_metric(y_true: List[float], y_pred: List[float]) -> float:
    """
    Spearman rank correlation.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        The Spearman rank correlation.
    """
    assert len(y_true) == len(y_pred)
    assert len(y_true) > 0
    res = scipy.stats.spearmanr(y_true, y_pred).correlation
    return float(res)


def mae_metric(y_true: List[float], y_pred: List[float]) -> float:
    """
    Mean absolute deviation.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        The mean absolute deviation.
    """
    assert len(y_true) == len(y_pred)
    assert len(y_true) > 0
    res = sum([abs(a - b) for (a, b) in zip(y_true, y_pred)]) / len(y_true)
    return float(res)


def mean_bias_metric(y_true: List[float], y_pred: List[float]) -> float:
    """
    Mean Bias.

    Bias is defined as the mean of the predicted values minus the true values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        The bias.
    """
    assert len(y_true) == len(y_pred)
    assert len(y_true) > 0
    res = sum([b - a for (a, b) in zip(y_true, y_pred)]) / len(y_true)
    return float(res)


def median_bias_metric(y_true: List[float], y_pred: List[float]) -> float:
    """
    Median Bias.

    Bias is defined as the mean of the predicted values minus the true values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        The bias.
    """
    assert len(y_true) == len(y_pred)
    assert len(y_true) > 0
    res = np.median([b - a for (a, b) in zip(y_true, y_pred)])
    return float(res)


def max_bias_metric(y_true: List[float], y_pred: List[float]) -> float:
    """
    Max Bias.

    Max Bias is defined as the max of the predicted values minus the true values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        The bias.
    """
    assert len(y_true) == len(y_pred)
    assert len(y_true) > 0
    res = max([b - a for (a, b) in zip(y_true, y_pred)])
    return float(res)


def min_bias_metric(y_true: List[float], y_pred: List[float]) -> float:
    """
    Min Bias.

    Min Bias is defined as the min of the predicted values minus the true values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        The bias.
    """
    assert len(y_true) == len(y_pred)
    assert len(y_true) > 0
    res = min([b - a for (a, b) in zip(y_true, y_pred)])
    return float(res)


def bias_binary_metric(y_true: List[float], y_pred: List[float]) -> float:
    """
    Bias binary.

    Bias binary is defined as the mean of:
        1.0 * [the predicted value > the true value] +
        0.5 * [the predicted value == the true value] +
        0.0 * [the predicted value < the true value]
    i.e. the perfect bias binary is 0.5. Overestimation of the true value
    results in bias binary in (0.5, 1.0], and underestimation in [0.0, 0.5).

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        The bias binary.
    """
    assert len(y_true) == len(y_pred)
    assert len(y_true) > 0
    res = sum(
        [
            (1.0 * (b > a) + 0.5 * (b == a) + 0.0 * (b < a))
            for (a, b) in zip(y_true, y_pred)
        ]
    ) / len(y_true)
    return float(res)


def bias_binary_symmetric_metric(
    y_true: List[float], y_pred: List[float]
) -> float:
    """
    Bias binary, but on the interval [-1, 1].

    Bias binary is defined as the mean of:
        1.0 * [the predicted value > the true value] +
        0.0 * [the predicted value == the true value] +
        -1.0 * [the predicted value < the true value]
    i.e. the perfect bias binary is 0.0. Overestimation of the true value
    results in bias binary in [0.0, 1.0], and underestimation in [-1.0, 0.0].

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.

    Returns:
        The bias binary.
    """
    assert len(y_true) == len(y_pred)
    assert len(y_true) > 0
    res = sum(
        [
            (1.0 * (b > a) + 0.0 * (b == a) - 1.0 * (b < a))
            for (a, b) in zip(y_true, y_pred)
        ]
    ) / len(y_true)
    return float(res)


def relative_error(y_true: List[float], y_pred: List[float]) -> float:
    """
    Relative error. The target should be only one real number.

    Args:
        y_true: Ground truth value.
        y_pred: Predicted value.

    Returns:
        The relative error.
    """
    assert len(y_pred) == 1
    assert len(y_true) == 1
    return float(abs(y_pred[0] - y_true[0]) / y_true[0])


def max_relative_error(y_true: List[float], y_pred: List[float]) -> float:
    """
    Max relative error. The target should be only one real number.

    Args:
        y_true: Ground truth value.
        y_pred: Predicted value.

    Returns:
        The relative error.
    """
    assert len(y_pred) == 1
    assert len(y_true) == 1
    return float((max(y_pred[0], y_true[0]) / min(y_pred[0], y_true[0])) - 1.0)


def log_relative_error(y_true: List[float], y_pred: List[float]) -> float:
    """
    Logarithmic relative error. The target should be only one real number.

    Args:
        y_true: Ground truth value.
        y_pred: Predicted value.

    Returns:
        The relative error.
    """
    assert len(y_pred) == 1
    assert len(y_true) == 1
    return float(abs(np.log(y_pred[0] / y_true[0])))


def get_metric_by_config(
    metric_config: str,
) -> Callable[[List[float], List[float]], float]:
    """
    Universal way to get a metric by name.
    """
    identifier, args = metric_config
    if len(args) > 0:
        raise ValueError(
            f"Metric {identifier} takes no arguments. You provided: {args}"
        )
    if identifier == "sr2":
        metric_fn = sr2_metric
    elif identifier == "mae":
        metric_fn = mae_metric
    elif identifier == "mean_bias":
        metric_fn = mean_bias_metric
    elif identifier == "median_bias":
        metric_fn = median_bias_metric
    elif identifier == "max_bias":
        metric_fn = max_bias_metric
    elif identifier == "min_bias":
        metric_fn = min_bias_metric
    elif identifier == "bias_binary":
        metric_fn = bias_binary_metric
    elif identifier == "bias_binary_symmetric":
        metric_fn = bias_binary_symmetric_metric
    elif identifier == "re":
        metric_fn = relative_error
    elif identifier == "mre":
        metric_fn = max_relative_error
    elif identifier == "lre":
        metric_fn = log_relative_error
    else:
        raise ValueError(f"Unknown metric name: {identifier}")
    return metric_fn
