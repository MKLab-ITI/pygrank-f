from pyfop import *
from pygrankf.core import backend, GraphSignal


@lazy
@autoaspects
def normalize(signal: GraphSignal, norm=None) -> GraphSignal:
    if norm is None:
        return signal/backend.max(signal)
    norm_value = backend.sum(signal**norm)**(1./norm)
    return signal / norm_value
