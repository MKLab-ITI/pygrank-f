import yaml
from typing import Union


def benchmark(settings: Union[str, dict], run):
    import pygrankf as pgf
    if not isinstance(settings, dict):
        with open(settings) as file:
            settings = yaml.load(file, Loader=yaml.SafeLoader)
    for setting in settings:
        pgf.print("\n"+"-"*10+" "+setting["name"]+" "+"-"*10+"\n")
        line = None
        for dataset in setting['datasets']:
            if dataset.get("skip", False):
                continue
            communities = pgf.load(dataset['name'],
                                   min_members=dataset.get('min_members', 0.01),
                                   num_groups=dataset.get('num_groups', 20))
            named_communities = {label: communities[name] for label, name in dataset['communities'].items()}
            hide_community_names = set(dataset['communities'].values())
            communities = {name: signal for name, signal in communities.items() if name not in hide_community_names}
            for split in setting['community']['splits']:
                if split.get("skip", "False") == "True":
                    continue
                for community_name, community in communities.items():
                    variables = {k: v for k, v in named_communities.items()}
                    pending_variables = dict()
                    for variable, fraction in split["variables"].items():
                        if fraction in variables:
                            variables[variable] = variables[fraction]
                        elif fraction == 'None':
                            variables[variable] = None
                        elif fraction == 'Random':
                            import random
                            variables[variable] = pgf.to_signal(community, {v: random.random() for v in community})
                        elif fraction == 'Remaining':
                            variables[variable] = community
                        else:
                            variables[variable], community = community.split(fraction, seed=split.get("seed", 0))
                    algorithms = run(**variables)
                    variables = variables | algorithms  # be able to reference specific algorithms in metrics
                    for variable, fraction in pending_variables:
                        variables[variable] = algorithms.get(fraction)
                    if not line:
                        line = [" "*30] + [algorithm for algorithm in algorithms for _ in setting['community']['metrics']]
                        pgf.print(*line)
                        line = [" "*30] + [metric["name"] for _ in algorithms for metric in setting['community']['metrics']]
                        pgf.print(*line)
                    line = [(dataset['name']+" "+str(community_name)+" ("+str(split["name"])+")").ljust(30)]
                    for algorithm, results in algorithms.items():
                        variables["run"] = results
                        for metric in setting['community']['metrics']:
                            func = getattr(pgf, metric["name"])
                            args = [variables[arg] for arg in metric["args"]]
                            line.append(func(*args))
                    pgf.print(*line)
