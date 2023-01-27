from pyfop import *
from pygrankf.core import backend, GraphSignal
from typing import Optional
from sklearn import metrics
from scipy import stats
import numpy as np


@eager_no_cache
def mabs(
    test: GraphSignal, prediction: GraphSignal, exclude: Optional[GraphSignal] = None
) -> float:
    assert test is not None
    prediction = prediction.filter(exclude)
    test = test.filter(exclude)
    return backend.sum(backend.abs(prediction - test)) / backend.length(test)


@eager_no_cache
def mmabs(
    test: GraphSignal, prediction: GraphSignal, exclude: Optional[GraphSignal] = None
) -> float:
    assert test is not None
    prediction = prediction.filter(exclude)
    test = test.filter(exclude)
    return backend.sum(backend.abs(prediction - test)) / backend.sum(test)


@eager_no_cache
def cos(
    test: GraphSignal, prediction: GraphSignal, exclude: Optional[GraphSignal] = None
) -> float:
    assert test is not None
    prediction = prediction.filter(exclude)
    test = test.filter(exclude)
    prediction = prediction / backend.sum(prediction**2) ** 0.5
    test = test / backend.sum(test**2) ** 0.5
    return backend.sum(prediction * test)


@eager_no_cache
def l1(
    test: GraphSignal, prediction: GraphSignal, exclude: Optional[GraphSignal] = None
) -> float:
    assert test is not None
    prediction = prediction.filter(exclude)
    test = test.filter(exclude)
    return backend.sum(backend.abs(prediction - test))


@eager_no_cache
def l1reg(prediction: GraphSignal, exclude: Optional[GraphSignal] = None) -> float:
    prediction = prediction.filter(exclude)
    return backend.sum(backend.abs(prediction))


@eager_no_cache
def l2(
    test: GraphSignal, prediction: GraphSignal, exclude: Optional[GraphSignal] = None
) -> float:
    assert test is not None
    prediction = prediction.filter(exclude)
    test = test.filter(exclude)
    return backend.sum(backend.abs(prediction - test) ** 2) ** 0.5


@eager_no_cache
def msq(
    test: GraphSignal, prediction: GraphSignal, exclude: Optional[GraphSignal] = None
) -> float:
    assert test is not None
    prediction = prediction.filter(exclude)
    test = test.filter(exclude)
    return backend.sum(backend.abs(prediction - test) ** 2)


@eager_no_cache
def l1partial(
    test: GraphSignal, prediction: GraphSignal, exclude: Optional[GraphSignal] = None
) -> float:
    assert test is not None
    prediction = prediction.filter(exclude)
    test = test.filter(exclude)
    return sgn(prediction - test)


def sgn(x):
    return x / (backend.abs(x) + 1.0e-12)


@eager_no_cache
def nce(
    test: GraphSignal, prediction: GraphSignal, exclude: Optional[GraphSignal] = None
) -> float:
    assert test is not None
    prediction = prediction.filter(exclude)
    prediction = prediction / backend.sum(prediction)
    test = test.filter(exclude)
    test = test / backend.sum(test)
    return backend.sum(backend.log(prediction + 1.0e-12) * test) / backend.sum(
        backend.log(test + 1.0e-12) * test
    )


@eager_no_cache
def auc(
    test: GraphSignal, prediction: GraphSignal, exclude: Optional[GraphSignal] = None
) -> float:
    assert test and prediction
    prediction = np.array(prediction.filter(exclude), copy=True)
    test = np.array(test.filter(exclude), copy=True)
    return metrics.roc_auc_score(test, prediction)


@eager_no_cache
def spearman(
    test: GraphSignal, prediction: GraphSignal, exclude: Optional[GraphSignal] = None
) -> float:
    assert test and prediction
    prediction = np.array(prediction.filter(exclude), copy=True)
    test = np.array(test.filter(exclude), copy=True)
    return stats.spearmanr(test, prediction)[0]


@eager_no_cache
def pearson(
    test: GraphSignal, prediction: GraphSignal, exclude: Optional[GraphSignal] = None
) -> float:
    assert test and prediction
    prediction = prediction.filter(exclude)
    test = test.filter(exclude)
    cov = backend.length(test) * backend.sum(test * prediction) - backend.sum(
        prediction
    ) * backend.sum(test)
    var_test = backend.length(test) * backend.sum(test**2) - backend.sum(test) ** 2
    var_pred = (
        backend.length(prediction) * backend.sum(prediction**2)
        - backend.sum(prediction) ** 2
    )
    return cov / var_test**0.5 / var_pred**0.5


@eager_no_cache
def prule(
    sensitive: GraphSignal,
    prediction: GraphSignal,
    exclude: Optional[GraphSignal] = None,
) -> float:
    assert sensitive and prediction
    prediction = prediction.filter(exclude)
    sensitive = sensitive.filter(exclude)
    sensitive_rate = backend.sum(sensitive * prediction) / backend.sum(sensitive)
    non_sensitive_rate = backend.sum((1 - sensitive) * prediction) / backend.sum(
        1 - sensitive
    )
    return min(sensitive_rate, non_sensitive_rate) / max(
        sensitive_rate, non_sensitive_rate
    )


@eager_no_cache
def cv(
    sensitive: GraphSignal,
    prediction: GraphSignal,
    exclude: Optional[GraphSignal] = None,
) -> float:
    assert sensitive and prediction
    prediction = prediction.filter(exclude)
    sensitive = sensitive.filter(exclude)
    sensitive_rate = backend.sum(sensitive * prediction) / backend.sum(sensitive)
    non_sensitive_rate = backend.sum((1 - sensitive) * prediction) / backend.sum(
        1 - sensitive
    )
    return backend.abs(sensitive_rate - non_sensitive_rate)
