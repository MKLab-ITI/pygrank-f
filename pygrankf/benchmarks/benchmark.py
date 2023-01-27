import yaml
from typing import Union


def benchmark(settings: Union[str, dict], run, **kwargs):
    import pygrankf as pgf
    if not isinstance(settings, dict):
        with open(settings) as file:
            settings = yaml.load(file, Loader=yaml.SafeLoader)
    for setting in settings:
        pgf.print("\n"+"-"*10+" "+setting["name"]+" "+"-"*10+"\n", **kwargs)
        line = None
        for dataset in setting['datasets']:
            summary = list() if dataset.get("summary", False) else None
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
                    #from pygrankf import backend
                    #print(backend.length(community))
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
                        pgf.print(*line, **kwargs)
                        line = [" "*30] + [metric["name"] for _ in algorithms for metric in setting['community']['metrics']]
                        pgf.print(*line, **kwargs)
                    line = [(dataset['name']+" "+str(community_name)+" ("+str(split["name"])+")").ljust(30)]
                    for algorithm, results in algorithms.items():
                        variables["run"] = results
                        for metric in setting['community']['metrics']:
                            func = getattr(pgf, metric["name"])
                            args = [variables[arg] for arg in metric["args"]]
                            line.append(func(*args))
                    if summary is not None:
                        if len(summary) == 0:
                            summary = [dataset['name'].ljust(30)] + [list() for _ in range(len(line)-1)]
                        for i in range(1, len(line)):
                            summary[i].append(line[i])
                    #else:
                    pgf.print(*line, **kwargs)
            if summary:
                pgf.print(*[sum(values)/len(values) if isinstance(values, list) else values for values in summary], **kwargs)
