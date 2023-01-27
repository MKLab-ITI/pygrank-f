import pygrankf as pgf


def run(train, exclude, sensitive, **kwargs):
    original = pgf.pagerank(train).call(spectrum="col")
    lfprp = pgf.lfprp(sensitive, original)(train).call()
    mult = pgf.fairmult(sensitive, exclude)(original)

    fp = pgf.steps(
        pgf.culep(train, original, sensitive),
        pgf.pagerank,
        pgf.tune(
                 metric=lambda *args: pgf.prule(sensitive, args[1], exclude)*2
                                      -pgf.l1(original, args[1], exclude)
                 )
    ).call(spectrum="col")

    gnn = pgf.steps(
        pgf.neural(train, original, sensitive),
        pgf.pagerank,
        pgf.tune(optimizer=pgf.tfsgd,
                 metric=lambda *args: pgf.prule(sensitive, args[1], exclude)*2
                                      -pgf.l1(mult, args[1], exclude)
                                      -pgf.sum(pgf.abs(args[1])*0.1)
                 )
    ).call(spectrum="col")
    return {"base": original, "mult": mult, "lfprp": lfprp, "fp": fp, "gnn": gnn}


pgf.benchmark("experiments/fairness/community.yaml", run, delim="&", endl="\\\\\n")
