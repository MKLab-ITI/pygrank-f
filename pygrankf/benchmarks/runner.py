import yaml
from typing import Union


def experiments(algorithms: Union[str, dict], **kwargs):
    def runner(**skwargs):
        return run(algorithms, **skwargs, **kwargs)
    return runner


def _parsearg(values, arg):
    import pygrankf as pgf
    if isinstance(arg, str):
        if hasattr(pgf, arg):
            return getattr(pgf, arg)
        return values[arg]
    generator = arg["generator"]
    if not callable(generator):
        generator = getattr(pgf, generator)
    if "args" in arg:
        return generator(*[a if a not in values else _parsearg(values, a) for a in arg["args"]])
    return generator(arg, values)


def metric(instructions: dict, values: dict):
    import pygrankf as pgf

    def compute(validation, predictions, exclude):
        assert validation is None
        assert exclude is None
        localvalues = values | {"run": predictions}
        result = 0
        for element in instructions["sum"]:
            func = getattr(pgf, element["name"])
            value = func(*[localvalues[arg] for arg in element["args"]])
            result = result + value*element.get("weight", 1)
        return result
    return compute


def run(algorithms: Union[str, dict], **kwargs):
    import pygrankf as pgf
    if not isinstance(algorithms, dict):
        with open(algorithms) as file:
            algorithms = yaml.load(file, Loader=yaml.SafeLoader)
    results = dict()
    values = {k: v for k, v in kwargs.items()}
    for alg in algorithms:
        steps = list()
        for step in alg["steps"]:
            func = getattr(pgf, step["name"]) if hasattr(pgf, step["name"]) else values[step["name"]]
            if "args" in step:
                func = func(*[_parsearg(values, arg) for arg in step["args"]])
            steps.append(func)
        result = pgf.steps(*steps)
        result = result.call(**{k: v if not isinstance(v, dict) and v not in values else _parsearg(values, v) for k, v in alg["aspects"].items()})
        results[alg["name"]] = result
        values[alg["name"]] = result
    return results
