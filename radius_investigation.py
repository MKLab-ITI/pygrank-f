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


def run(train, exclude, sensitive, **kwargs):
    coefficients = pgf.PageRank(0.8)
    original = pgf.steps(train, pgf.filter, pgf.normalize).call(
        spectrum="symmetric", norm=1, coefficients=coefficients
    )

    def create_metric(reg=0.1):
        def metric(*args):
            return (
                pgf.prule(sensitive, args[1], exclude) * 2
                - pgf.l1(original, args[1], exclude)
                - reg * pgf.sum(pgf.abs(args[1]))
            )
        return metric

    nn = argmax(
        lambda layers, reg: pgf.steps(
            pgf.neural(train, original, sensitive, layers=layers),
            pgf.normalize,
            pgf.tune(optimizer=pgf.tfsgd, metric=create_metric(reg)),
        ).call(norm=1),
        lambda predictions: create_metric(0)(None, predictions, None),
        product([2, 3, 4], [0.1]),
    )
    """
    gnn_wrong = pgf.steps(
        pgf.neural(train, original, sensitive),
        pgf.filter,
        pgf.normalize,
        pgf.tune(optimizer=pgf.tfsgd, metric=loss)
    ).call(norm=1, spectrum="symmetric", coefficients=pgf.PageRank(0.9))


    noreg = pgf.steps(
        pgf.neural(train, original, sensitive),
        pgf.filter,
        pgf.normalize,
        pgf.tune(optimizer=pgf.tfsgd, metric=loss)
    ).call(norm=1, spectrum="symmetric", coefficients=pgf.HeatKernels(7))
    """

    gnn = argmax(
        lambda layers, reg: pgf.steps(
            pgf.neural(train, original, sensitive, layers=layers),
            pgf.filter,
            pgf.normalize,
            pgf.tune(optimizer=pgf.tfsgd, metric=create_metric(reg)),
        ).call(norm=1, spectrum="symmetric", coefficients=coefficients),
        lambda predictions: create_metric(0)(None, predictions, None),
        product([2, 3, 4], [0.1]),
    )

    return {"base": original, "nn": nn, "gnn": gnn}


pgf.benchmark("experiments/fairness/test.yaml", run, delim="&", endl="\\\\\n")
