from pygrankf.measures import auc
from pygrankf.algorithms.autotune.optimizers import nonconvex


class Tunable:
    def __init__(self, mins, maxs, start=None):
        if not isinstance(mins, list):
            mins = [mins]
        if not isinstance(maxs, list):
            maxs = [maxs]
        assert len(mins) == len(maxs)
        self.mins = mins
        self.maxs = maxs
        self.start = [(mm+mx)*0.5 for mm, mx in zip(mins, maxs)] if start is None else start


def tune(validation=None, metric=auc, exclude=None, optimizer=nonconvex, **extrakwargs):
    class PendingTuner:
        def __init__(self, result):
            self.result = result

        def call(self, tuneparams={"deviation_tol": 1.E-6, "randomize_search": True}, **kwargs):
            for arg in kwargs:
                if arg in extrakwargs:
                    raise Exception("Argument declared twice for tuning:", arg)
            kwargs = kwargs | extrakwargs | self.result.get_input_context().values
            fixedkwargs = {k: v for k, v in kwargs.items() if not isinstance(v, Tunable)}
            kwargs = {k: v for k, v in kwargs.items() if isinstance(v, Tunable)}
            result = self.result
            mins = list()
            maxs = list()
            start = list()

            for limits in kwargs.values():
                mins = mins + limits.mins
                maxs = maxs + limits.maxs
                start = start + limits.start

            def params2kwargs(params):
                param_kwargs = dict()
                pos = 0
                for arg, limits in kwargs.items():
                    npos = pos + len(limits.mins)
                    param_kwargs[arg] = params[pos:npos]
                    pos = npos
                param_kwargs = {k: v if len(v) != 1 else v[0] for k, v in param_kwargs.items()}
                return param_kwargs

            def loss(params):
                predictions = result.call(**(params2kwargs(params) | fixedkwargs))
                return -metric(validation, predictions, exclude)
            bestparams = optimizer(loss, maxs, mins, start, **tuneparams)
            ret = result.call(**(params2kwargs(bestparams) | fixedkwargs))
            return ret
    return PendingTuner
