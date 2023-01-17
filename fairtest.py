import pygrankf as pgf

communities = pgf.load("polbooks")
print(communities.keys())
sensitive_community = 2
sensitive = communities[sensitive_community]

pgf.print("", "ppr", "ppr", "ppr", "|", "lfpro", "lfpro", "lfpro", "|", "fairedit", "fairedit", "fairedit")
pgf.print("", "auc", "utility", "prule", "|", "auc", "utility", "prule", "|", "auc", "utility", "prule")
for name, community in communities.items():
    if name == sensitive_community:
        continue
    train, test = community.split(0.5)
    exclude = None  # None or train
    original = pgf.pagerank(train).call(spectrum="col")
    lfpro = pgf.lfpro(train, sensitive, original).call()
    gnn = pgf.steps(
        pgf.neural(train, original, sensitive),
        pgf.pagerank,
        pgf.tune(optimizer=pgf.tfsgd,
                metric=lambda *args: min(pgf.prule(args[1], sensitive, exclude), 1)-pgf.l1(args[1], original, exclude))
    ).call(spectrum="col")

    pgf.print(name,
             pgf.auc(test, original, exclude),
             pgf.l1(original, original, exclude),
             pgf.prule(sensitive, original, exclude),
             "|",
             pgf.auc(test, lfpro, exclude),
             pgf.l1(original, lfpro, exclude),
             pgf.prule(sensitive, lfpro, exclude),
             "|",
             pgf.auc(test, gnn, exclude),
             pgf.l1(original, gnn, exclude),
             pgf.prule(sensitive, gnn, exclude))
