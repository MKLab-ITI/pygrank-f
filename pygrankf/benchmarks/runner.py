from typing import Union
from pygrankf.benchmarks.benchmark import _loadyaml, _str2list


def experiments(algorithms: Union[str, dict], **kwargs):
    def runner(**skwargs):
        return run(algorithms, **skwargs, **kwargs)

    return runner


def _parsearg(values, arg, update=False):
    if isinstance(arg, str) and arg in values:
        arg = values[arg]
    if isinstance(arg, dict) and arg.get("type", "None") == "dict":
        return arg
    if not isinstance(arg, str) and not isinstance(arg, dict):
        return arg
    if isinstance(arg, str) and ".yaml/" in arg:
        file, arg = arg.split(".yaml/")
        arg = _loadyaml(file + ".yaml", update)[arg]
    import pygrankf as pgf

    if isinstance(arg, str):
        if hasattr(pgf, arg):
            return getattr(pgf, arg)
        return values[arg]
    generator = arg["generator"]
    if not callable(generator):
        generator = getattr(pgf, generator)
    if "args" in arg or "kwargs" in arg:
        return generator(
            *[
                a if a not in values else _parsearg(values, a, update)
                for a in _str2list(arg.get("args", []))
            ],
            **{
                k: a if a not in values else _parsearg(values, a, update)
                for k, a in arg.get("kwargs", {}).items()
            }
        )
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
            value = func(
                *[localvalues[arg] for arg in _str2list(element.get("args", []))],
                **{k: localvalues[arg] for k, arg in element.get("kwargs", {}).items()}
            )
            if "min" in element:
                value = max(value, element["min"])
            if "max" in element:
                value = min(value, element["max"])
            result = result + value * element.get("weight", 1)
        return result

    return compute


def _runalg(alg, pgf, values, update):
    # runs algorithm once
    steps = list()
    for step in alg["steps"]:
        func = (
            getattr(pgf, step["name"])
            if hasattr(pgf, step["name"])
            else values[step["name"]]
        )
        if "args" in step or "kwargs" in step:
            func = func(
                *[_parsearg(values, arg, update) for arg in _str2list(step.get("args", []))],
                **{
                    k: _parsearg(values, arg, update)
                    for k, arg in step.get("kwargs", {}).items()
                }
            )
        steps.append(func)
    result = pgf.steps(*steps)
    result = result.call(
        **{
            k: v
            if not isinstance(v, dict)
               and v not in values
               and (not isinstance(v, str) or ".yaml" not in v)
            else _parsearg(values, v, update)
            for k, v in alg["aspects"].items()
        }
    )
    return result


def run(algorithms: Union[str, dict], update=False, **kwargs):
    import pygrankf as pgf

    if not isinstance(algorithms, dict):
        algorithms = _loadyaml(algorithms, update)
    results = dict()
    values = {k: v for k, v in kwargs.items()}
    for alg in algorithms:
        try:
            if "default" in alg:
                if alg["name"] not in values:
                    values[alg["name"]] = alg["default"]
                    if "get" in alg:
                        values[alg["default"]] = getattr(values[alg["get"].split(".")[0]], alg["get"].split(".")[1])
                continue
            if alg["name"] in values:
                continue
            if "spawn" in alg:
                metric = _parsearg(values, alg["spawn"]["select"], update)
                hyperparameters = {k: _str2list(v) for k, v in alg["spawn"]["hyperparameters"].items()}
                searchparameters = alg["spawn"].get("search", {})
                finalparameters = alg["spawn"].get("final", {})
                result = None
                best_value = float("-inf")
                best_params = None
                from itertools import product
                for hyperparameter_values in product(*list(hyperparameters.values())):
                    hyperparameter_kwargs = {k: v for k, v in zip(hyperparameters.keys(), hyperparameter_values)}
                    pgf.utils.prefix(alg["name"]+" "+str(searchparameters)+" "+str(hyperparameter_kwargs)+" ")
                    tmp_result = _runalg(alg, pgf, values | hyperparameter_kwargs | searchparameters, update)
                    value = metric(None, tmp_result, None)
                    if value > best_value:
                        best_value = value
                        best_params = hyperparameter_kwargs
                        result = tmp_result
                if finalparameters or finalparameters:
                    pgf.utils.prefix(alg["name"]+" "+str(finalparameters)+" "+str(best_params)+" ")
                    result = _runalg(alg, pgf, values | best_params | finalparameters, update)
                pgf.utils.prefix()
            else:
                pgf.utils.prefix(alg["name"] + " ")
                result = _runalg(alg, pgf, values, update)
                pgf.utils.prefix()
            values[alg["name"]] = result
            if alg.get("show", "True") == "True":
                results[alg["name"]] = result
        except Exception as e:
            import traceback
            print("Error during execution of "+str(alg["name"])+": "+str(e))
            traceback.print_exc()
    return results
