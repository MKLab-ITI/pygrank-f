import yaml
from typing import Union
import os
import wget


def _loadyaml(path, update=False):
    if path.startswith("https://"):
        download_path = os.path.join(
            os.path.expanduser("~"), ".pygrank/experiments", path[8:]
        )
        os.makedirs(os.path.split(download_path)[0], exist_ok=True)
        if not os.path.exists(download_path) or update:
            wget.download(path, download_path)
        path = download_path
    with open(path, "r") as file:
        ret = yaml.load(file, Loader=yaml.SafeLoader)
    return ret


def characteristics(settings: Union[str, dict], update=False, **kwargs):
    import pygrankf as pgf
    if not isinstance(settings, dict):
        settings = _loadyaml(settings, update)
    pgf.print("Community".ljust(30), "Nodes", "Edges", "Members", "Protected")
    for setting in settings:
        for dataset in setting["datasets"]:
            if dataset.get("skip", False):
                continue
            communities = pgf.load(
                dataset["name"],
                min_members=dataset.get("min_members", 0.01),
                num_groups=dataset.get("num_groups", 20),
            )
            named_communities = {
                label: communities[name]
                for label, name in dataset["communities"].items()
            }
            hide_community_names = set(dataset["communities"].values())
            communities = {
                name: signal
                for name, signal in communities.items()
                if name not in hide_community_names
            }
            for community_name, community in communities.items():
                name = (
                            dataset["name"]
                            + " "
                            + str(community_name)
                        ).ljust(30)
                pgf.print(name,
                          community.graph.number_of_nodes(),
                          community.graph.number_of_edges(),
                          pgf.sum(community),
                          pgf.sum(next(iter(named_communities.values()))),
                          **kwargs)


def benchmark(settings: Union[str, dict], run, update=False, total=False, **kwargs):
    import pygrankf as pgf

    if isinstance(run, str) or isinstance(run, dict):
        run = pgf.experiments(run, update=update)
    if not isinstance(settings, dict):
        settings = _loadyaml(settings, update)
    total = list() if total else None
    for setting in settings:
        pgf.print(
            "\n" + "-" * 10 + " " + setting["name"] + " " + "-" * 10 + "\n"
        )
        line = None
        for dataset in setting["datasets"]:
            summary = list() if dataset.get("summary", False) else None
            if dataset.get("skip", False):
                continue
            communities = pgf.load(
                dataset["name"],
                min_members=dataset.get("min_members", 0.01),
                num_groups=dataset.get("num_groups", 20),
            )
            named_communities = {
                label: communities[name]
                for label, name in dataset["communities"].items()
            }
            hide_community_names = set(dataset["communities"].values())
            communities = {
                name: signal
                for name, signal in communities.items()
                if name not in hide_community_names
            }
            for split in setting["community"]["splits"]:
                if split.get("skip", "False") == "True":
                    continue
                for community_name, community in communities.items():
                    # from pygrankf import backend
                    # print(backend.length(community))
                    variables = {k: v for k, v in named_communities.items()}
                    pending_variables = dict()
                    for variable, fraction in split["variables"].items():
                        if fraction in variables:
                            variables[variable] = variables[fraction]
                        elif fraction == "None":
                            variables[variable] = None
                        elif fraction == "RandomPos":
                            import random

                            variables[variable] = pgf.to_signal(
                                community,
                                {
                                    v: random.random() for v in community
                                },  # this is for all graph nodes
                            )
                        elif fraction == "Random":
                            import random

                            variables[variable] = pgf.to_signal(
                                community,
                                {
                                    v: 2 * (random.random() - 0.5) for v in community
                                },  # this is for all graph nodes
                            )
                        elif fraction == "Remaining":
                            variables[variable] = community
                        else:
                            variables[variable], community = community.split(
                                fraction, seed=split.get("seed", 0)
                            )
                    algorithms = run(**variables)
                    variables = (
                        variables | algorithms
                    )  # be able to reference specific algorithms in metrics
                    for variable, fraction in pending_variables:
                        variables[variable] = algorithms.get(fraction)
                    if not line:
                        line = [" " * 30] + [
                            algorithm
                            for algorithm in algorithms
                            for _ in setting["community"]["metrics"]
                        ]
                        pgf.print(*line, **kwargs)
                        line = [" " * 30] + [
                            metric["name"]
                            for _ in algorithms
                            for metric in setting["community"]["metrics"]
                        ]
                        pgf.print(*line, **kwargs)
                    line = [
                        (
                            dataset["name"]
                            + " "
                            + str(community_name)
                            + " ("
                            + str(split["name"])
                            + ")"
                        ).ljust(30)
                    ]
                    for algorithm, results in algorithms.items():
                        variables["run"] = results
                        for metric in setting["community"]["metrics"]:
                            func = getattr(pgf, metric["name"])
                            args = [variables[arg] for arg in metric["args"]]
                            line.append(func(*args))
                    if total is not None and summary is None:
                        if len(total) == 0:
                            total = ["total".ljust(30)] + [
                                list() for _ in range(len(line) - 1)
                            ]
                        for i in range(1, len(line)):
                            total[i].append(line[i])
                    if summary is not None:
                        if len(summary) == 0:
                            summary = [dataset["name"].ljust(30)] + [
                                list() for _ in range(len(line) - 1)
                            ]
                        for i in range(1, len(line)):
                            summary[i].append(line[i])
                    # else:
                    if dataset.get("show", "True") == "True":
                        pgf.print(*line, **kwargs)
            if summary:
                pgf.print(
                    *[
                        sum(values) / len(values)
                        if isinstance(values, list)
                        else values
                        for values in summary
                    ],
                    **kwargs
                )
                if total is not None:
                    if len(total) == 0:
                        total = ["total".ljust(30)] + [
                            list() for _ in range(len(summary) - 1)
                        ]
                    for i in range(1, len(summary)):
                        total[i].append(sum(summary[i])/len(summary[i]))
    if total:
        pgf.print(
            *[
                sum(values) / len(values)
                if isinstance(values, list)
                else values
                for values in total
            ],
            **kwargs
        )
