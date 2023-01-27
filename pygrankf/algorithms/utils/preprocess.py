import scipy
import numpy as np
import networkx as nx
from pygrankf.core import backend
from pyfop import *


@lazy
@autoaspects
def reweigh(graph, spectrum="auto", weight="weight", renormalize=False, cors=True):
    normalization = spectrum.lower()
    if normalization == "auto":
        normalization = "col" if graph.is_directed() else "symmetric"
    #M = nx.to_scipy_sparse_matrix(graph, weight=weight, dtype=float)
    M = graph.to_scipy_sparse_array()
    if weight != "weight":
        raise Exception("Fastgraph does not support custom edge weights")
    if renormalize:
        M = M + scipy.sparse.eye(M.shape[0])*float(renormalize)
    if normalization == "col":
        S = np.array(M.sum(axis=1)).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
        M = Q * M
    elif normalization == "symmetric":
        S = np.array(np.sqrt(M.sum(axis=1))).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Qleft = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
        S = np.array(np.sqrt(M.sum(axis=0))).flatten()
        S[S != 0] = 1.0 / S[S != 0]
        Qright = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
        M = Qleft * M * Qright
    elif normalization != "none":
        raise Exception("Supported normalizations: none, col, symmetric, auto")
    ret = backend.scipy_sparse_to_backend(M)
    if cors:
        ret.cors = {"numpy": M, backend.backend_name(): ret}
    return ret
