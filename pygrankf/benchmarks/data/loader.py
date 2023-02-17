import os
from pyfop import *
from pygrankf.core import to_signal, Fastgraph, utils
from pygrankf.benchmarks.data import resources


@eager_no_cache
@autoaspects
def load(
    dataset: str,
    path: str = Aspect(),
    pairs: str = "pairs.txt",
    groups: str = "groups.txt",
    groupnames: str = "groupnames.txt",
    directed: bool = False,
    min_members: float = 0.01,
    num_groups: int = 20,
    subset: int = None,
    download=resources.download,
):
    download(dataset)
    # G = nx.DiGraph() if directed else nx.Graph()
    utils.log(f"Loading {dataset} graph")
    G = Fastgraph(directed=directed)
    groupdict = {}
    subset = float('inf') if subset is None else subset
    with open(path + "/" + dataset + "/" + pairs, "r", encoding="utf-8") as file:
        for line in file:
            if len(line) != 0 and line[0] != "#":
                splt = line[:-1].split()
                if len(splt) > 1:
                    G.add_edge(splt[0], splt[1])
                subset -= 1
                if subset <= 0:
                    break
    if min_members < 1:
        min_members *= len(G)

    utils.log(f"Loading {dataset} communities")
    groupnameslist = None
    if groupnames is not None and os.path.isfile(
        os.path.join(path, dataset, groupnames)
    ):
        with open(
            os.path.join(path, dataset, groupnames), "r", encoding="utf-8"
        ) as file:
            groupnameslist = [line[:-1] for line in file]

    if groups is not None and os.path.isfile(os.path.join(path, dataset, groups)):
        with open(os.path.join(path, dataset, groups), "r", encoding="utf-8") as file:
            for groupid, line in enumerate(file):
                if line[0] == "#":
                    continue
                group = [
                    item for item in line[:-1].split() if len(item) > 0 and item in G
                ]
                if len(group) >= min_members:
                    groupdict[
                        groupid if groupnameslist is None else groupnameslist[groupid]
                    ] = group
                    if len(groupdict) >= num_groups:
                        break
    utils.log()
    return {k: to_signal(G, v) for k, v in groupdict.items()}
