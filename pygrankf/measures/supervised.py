from pyfop import *
from pygrankf.core import backend, GraphSignal
from typing import Optional
from sklearn import metrics
import numpy as np


def aggregate(kwargs):
    @eager_no_cache
    def combined(test: Optional[GraphSignal] = None, prediction: Optional[GraphSignal] = None, exclude: Optional[GraphSignal] = None):
        result = 0
        for metric, weight in kwargs.items():
            result = result + weight*metric(test, prediction, exclude)
            return result
    return combined


@eager_no_cache
def mabs(test: GraphSignal, prediction: GraphSignal, exclude: Optional[GraphSignal] = None) -> float:
    assert test is not None
    prediction = prediction.filter(exclude)
    test = test.filter(exclude)
    return backend.sum(backend.abs(prediction-test))/backend.length(test)


@eager_no_cache
def l1(test: GraphSignal, prediction: GraphSignal, exclude: Optional[GraphSignal] = None) -> float:
    assert test is not None
    prediction = prediction.filter(exclude)
    test = test.filter(exclude)
    return backend.sum(backend.abs(prediction-test))


@eager_no_cache
def ce(test: GraphSignal, prediction: GraphSignal, exclude: Optional[GraphSignal] = None) -> float:
    assert test is not None
    prediction = prediction.filter(exclude)
    prediction = prediction / backend.sum(prediction)
    test = test.filter(exclude)
    test = test/backend.sum(test)
    return -backend.sum(backend.log(prediction+1.E-6)*test)+backend.sum(backend.log(test+1.E-6)*test)


@eager_no_cache
def auc(test: GraphSignal, prediction: GraphSignal, exclude: Optional[GraphSignal] = None) -> float:
    assert test and prediction
    prediction = np.array(prediction.filter(exclude), copy=False)
    test = np.array(test.filter(exclude), copy=False)
    fpr, tpr, thresholds = metrics.roc_curve(test, prediction, pos_label=1)
    return metrics.auc(fpr, tpr)


@eager_no_cache
def prule(sensitive: GraphSignal, prediction: GraphSignal, exclude: Optional[GraphSignal] = None) -> float:
    assert sensitive and prediction
    prediction = prediction.filter(exclude)
    sensitive = sensitive.filter(exclude)
    sensitive_rate = backend.sum(sensitive*prediction) / backend.sum(sensitive)
    non_sensitive_rate = backend.sum((1-sensitive)*prediction) / backend.sum(1-sensitive)
    return min(sensitive_rate, non_sensitive_rate) / max(sensitive_rate, non_sensitive_rate)
