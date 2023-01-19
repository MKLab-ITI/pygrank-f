from pyfop import *
from pygrankf.core import backend, GraphSignal


@lazy_no_cache
@autoaspects
def normalize(signal: GraphSignal, norm=None) -> GraphSignal:
    if norm is None:
        return signal/backend.max(signal)
    norm_value = backend.sum(signal**norm)**(1./norm)
    return signal / norm_value


def sweep(original: GraphSignal):
    @lazy_no_cache
    @autoaspects
    def ratio(signal: GraphSignal, original: GraphSignal) -> GraphSignal:
        return signal / original

    def method(signal):
        return ratio(signal, original)
    return method
