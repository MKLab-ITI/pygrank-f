from pyfop import *
from pygrankf import backend
from pygrankf.core import GraphSignal
from pygrankf.algorithms.utils import Convergence, reweigh
from pygrankf.algorithms.autotune.autogf import Tunable
from typing import Iterable


@lazy
@autoaspects
def filter(personalization: GraphSignal, coefficients: Iterable[float] = Tunable([0]*40, [1]*40),
           normalize=reweigh, convergence=Convergence()) -> GraphSignal:
    coefficients = iter(coefficients)
    normalized_adjacency = normalize(personalization.graph)
    result = personalization*next(coefficients)
    power = personalization
    convergence.start()
    while not convergence.has_converged(result):
        power = backend.conv(power, normalized_adjacency)
        try:
            result = result + next(coefficients)*power
        except StopIteration:
            break
    return result
