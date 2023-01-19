from pyfop import *
from pygrankf import backend
from pygrankf.algorithms.utils.convergence import Convergence
from pygrankf.algorithms.utils.preprocess import reweigh
from typing import Iterable


class PageRank(Iterable[float]):
    def __init__(self, alpha=0.85):
        self.alpha = alpha

    def __iter__(self):
        class Iterator:
            def __init__(self, alpha):
                self.alpha = alpha
                self.product = (1-alpha)

            def __next__(self):
                ret = self.product
                self.product *= self.alpha
                return ret
        return Iterator(self.alpha)


@lazy_no_cache
@autoaspects
def pagerank(personalization, alpha=0.85, normalize=reweigh, convergence=Convergence()):
    normalized_adjacency = normalize(personalization.graph)
    personalization = personalization/backend.sum(personalization)
    result = personalization
    convergence.start()
    while not convergence.has_converged(result):
        result = personalization*(1-alpha) + alpha*backend.conv(personalization, normalized_adjacency)
        result = result / backend.sum(result)
    return result
