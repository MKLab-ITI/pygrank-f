import pygrankf as pgf
from itertools import product


def argmax(func, metric, options):
    best_value = float("-inf")
    best_result = None
    for option in options:
        result = func(*option)
        value = metric(result)
        if value > best_value:
            best_value = value
            best_result = result
    return best_result

"""
def run(train, exclude, sensitive, **kwargs):
    coefficients = pgf.HeatKernels(1)
    original = pgf.steps(train, pgf.filter, pgf.normalize).call(
        spectrum="symmetric", norm=1, coefficients=coefficients
    )
    hyperparams = list(product([2, 3, 4], [2, 5]))

    def create_metric(reg=None, regl1=1):
        def metric(*args):
            if reg is None:
                return float("-inf") if pgf.prule(sensitive, args[1], exclude) < 0.99 else - pgf.l1(original, args[1], exclude)
            return (
                -(1-pgf.prule(sensitive, args[1], exclude)) * reg
                - pgf.l1(original, args[1], exclude)
                - regl1*pgf.mean(pgf.abs(args[1]))
            )
        return metric

    nn = argmax(
        lambda layers, reg: pgf.steps(
            pgf.neural(train, original, sensitive, layers=layers),
            pgf.normalize,
            pgf.tune(optimizer=pgf.tfsgd, metric=create_metric(reg)),
        ).call(norm=1),
        lambda predictions: create_metric()(None, predictions, None),
        hyperparams,
    )
    wrong = argmax(
        lambda layers, reg: pgf.steps(
            pgf.neural(train, original, sensitive, layers=layers),
            pgf.filter,
            pgf.normalize,
            pgf.tune(optimizer=pgf.tfsgd, metric=create_metric(reg)),
        ).call(norm=1, spectrum="symmetric", coefficients=pgf.PageRank(0.9)),
        lambda predictions: create_metric()(None, predictions, None),
        hyperparams,
    )

    noreg = argmax(
        lambda layers, reg: pgf.steps(
            pgf.neural(train, original, sensitive, layers=layers),
            pgf.filter,
            pgf.normalize,
            pgf.tune(optimizer=pgf.tfsgd, metric=create_metric(reg, 0)),
        ).call(norm=1, spectrum="symmetric", coefficients=coefficients),
        lambda predictions: create_metric()(None, predictions, None),
        hyperparams,
    )

    gnn = argmax(
        lambda layers, reg: pgf.steps(
            pgf.neural(train, original, sensitive, layers=layers),
            pgf.filter,
            pgf.normalize,
            pgf.tune(optimizer=pgf.tfsgd, metric=create_metric(reg)),
        ).call(norm=1, spectrum="symmetric", coefficients=coefficients),
        lambda predictions: create_metric()(None, predictions, None),
        hyperparams,
    )

    return {"base": original, "nn": nn, "noreg": noreg, "wrong": wrong, "gnn": gnn}
"""


#experiments = pgf.experiments("experiments/algorithms/ablation.yaml")
#pgf.benchmark("experiments/fairness/test.yaml", experiments, delim="&", endl="\\\\\n")

pgf.benchmark("https://raw.githubusercontent.com/maniospas/pygrank-f/main/experiments/fairness/test.yaml",
              "experiments/algorithms/ablation.yaml",
              update=True,
              delim="&", endl="\\\\\n")
