import pygrankf as pgf


def run(train, exclude, sensitive, **kwargs):
    original = pgf.pagerank(train).call(spectrum="col")
    lfpro = pgf.lfpro(sensitive, original)(train).call()
    #return {"base": original, "lfpro": lfpro}

    gnn = pgf.steps(
        pgf.neural(train, original, sensitive),
        pgf.pagerank,
        pgf.tune(optimizer=pgf.tfsgd,
                 metric=lambda *args: pgf.prule(sensitive, args[1], exclude)
                                      -pgf.l1(original, args[1], exclude)*0.5
                                      -pgf.sum(args[1])*0.5
                 )
    ).call(spectrum="col")
    return {"base": original, "lfpro": lfpro, "gnn": gnn}


pgf.benchmark("experiments/fairness/community.yaml", run)
