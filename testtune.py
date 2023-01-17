import pygrankf as pgf

communities = pgf.load("citeseer")
for name, community in communities.items():
    train, test = community.split(0.5)
    train, validation = train.split(0.9)
    result = pgf.steps(
        train,
        pgf.filter,
        pgf.tune(validation=validation, metric=pgf.mabs, exclude=train)
    ).call(coefficients=pgf.Tunable([0]*40, [1]*40))
    pgf.print(name, pgf.auc(test, result))
