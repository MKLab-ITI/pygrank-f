from pyfop import *
from pygrankf.core import backend, GraphSignal, to_signal
from typing import Union


@lazy_no_cache
@autoaspects
def normalize(signal: GraphSignal, norm=None) -> GraphSignal:
    if norm is None:
        return signal / backend.max(signal)
    norm_value = backend.sum(signal**norm) ** (1.0 / norm)
    return signal / norm_value


def sweep(original: GraphSignal, desired: Union[float, GraphSignal] = 1.0):
    @lazy_no_cache
    @autoaspects
    def ratio(
        signal: GraphSignal,
        original: GraphSignal,
        desired: Union[float, GraphSignal],
        sweep_offset: Union[float, str] = 0,
    ) -> GraphSignal:
        if sweep_offset == "max":
            sweep_offset = backend.max(original)
        if sweep_offset < 0:
            raise Exception("Negative sweep offset")
        return (signal + sweep_offset) * desired / (original + sweep_offset)

    def method(signal):
        return ratio(signal, original, desired)

    return method


def fairmult(sensitive: GraphSignal, exclude: GraphSignal = None):
    @lazy_no_cache
    @autoaspects
    def method(original: GraphSignal, prule: float = 1):
        s = sensitive.filter(exclude)
        sum_sensitive = backend.sum(original * s)
        sum_non_sensitive = backend.sum(original * (1 - s))
        if sum_non_sensitive == 0 or sum_sensitive == 0:
            return original
        mean_sensitive = sum_sensitive / backend.sum(s)
        mean_non_sensitive = sum_non_sensitive / backend.sum(1 - s)
        ret = sensitive * original * prule + (1 - sensitive) * original * (
            mean_sensitive / mean_non_sensitive
        )
        ret = ret / backend.sum(backend.abs(ret)) * backend.sum(backend.abs(original))
        return ret

    return method


def lfpro(sensitive: GraphSignal, exclude: GraphSignal = None):
    def distribute(
        DR: GraphSignal, ranks: GraphSignal, sensitive: GraphSignal, eps: float
    ):
        min_rank = float("inf")
        while min_rank >= eps and DR > 0:
            ranks = {
                v: ranks[v] * sensitive.get(v, 0)
                for v in ranks
                if ranks[v] * sensitive.get(v, 0) != 0
            }
            d = DR / len(ranks)
            min_rank = min(ranks.values())
            if min_rank > d:
                min_rank = d
            ranks = {v: val - min_rank for v, val in ranks.items()}
            DR -= len(ranks) * min_rank
        return ranks

    @lazy_no_cache
    @autoaspects
    def method(ranks: GraphSignal, eps=1.0e-12):
        phi = backend.sum(sensitive) / backend.length(sensitive)
        ranks = ranks / backend.sum(ranks)
        sumR = backend.sum(ranks * sensitive)
        sumB = backend.sum(ranks * (1 - sensitive))
        numR = backend.sum(sensitive)
        numB = backend.length(ranks) - numR
        if sumR < phi:
            red = distribute(
                phi - sumR, ranks, {v: 1 - sensitive.get(v, 0) for v in ranks}, eps
            )
            ranks = to_signal(
                ranks, {v: red.get(v, ranks[v] + (phi - sumR) / numR) for v in ranks}
            )
        elif sumB < 1 - phi:
            red = distribute(
                1 - phi - sumB, ranks, {v: sensitive.get(v, 0) for v in ranks}, eps
            )
            ranks = to_signal(
                ranks,
                {v: red.get(v, ranks[v] + (1 - phi - sumB) / numB) for v in ranks},
            )
        return ranks

    return method
