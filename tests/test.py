import pygrankf as pgf


communities = pgf.load("citeseer")
for name, community in communities.items():
    train, test = community.split(0.5)
    train, validation = train.split(0.9)
    algorithm = pgf.pagerank(train)
    result = algorithm.call()
    pgf.print(name, pgf.auc(test, result))
