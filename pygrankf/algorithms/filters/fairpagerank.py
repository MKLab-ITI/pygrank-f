from pyfop import *
from pygrankf import backend
from pygrankf.algorithms.utils import Convergence
import scipy.sparse


def safe_div(a, b):
    if b == 0:
        return 0
    return a / b


@lazy_no_cache
@autoaspects
def lfpro(personalization, sensitive, original, alpha=0.85, convergence=Convergence()):
    phi = backend.sum(sensitive) / backend.length(sensitive)
    # custom graph preprocessing
    M = personalization.graph.to_scipy_sparse_array()
    outR = backend.conv(sensitive.np, M)
    outB = backend.conv(1. - sensitive.np, M)
    case1 = (outR < (phi * (outR + outB)))
    case2 = ((1 - case1) * (outR != 0))
    case3 = ((1 - case1) * (1 - case2))
    d = case1 * backend.safe_inv(outB) * (1 - phi) + case2 * backend.safe_inv(outR) * phi + case3
    Q = scipy.sparse.spdiags(d, 0, *M.shape)
    M = Q @ M
    normalized_adjacency = backend.scipy_sparse_to_backend(M)

    #
    dR = case1 * (phi - (1 - phi) * backend.safe_inv(outB) * outR) + case3 * phi
    dB = case2 * ((1 - phi) - phi * backend.safe_inv(outR) * outB) + case3 * (1 - phi)
    xR = safe_div(original * sensitive, backend.sum(original * sensitive))
    xB = safe_div(original * (1 - sensitive), backend.sum(original * (1 - sensitive)))

    personalization = personalization / backend.sum(personalization)
    result = personalization
    convergence.start()
    while not convergence.has_converged(result):
        deltaR = backend.sum(result * dR)
        deltaB = backend.sum(result * dB)
        result = (backend.conv(result, normalized_adjacency) + xR*deltaR + xB * deltaB) * alpha + personalization * (1 - alpha)
        result = result / backend.sum(result)
    return result - personalization*(1-alpha)
