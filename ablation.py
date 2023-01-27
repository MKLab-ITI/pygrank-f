import pygrankf as pgf


#experiments = pgf.experiments("experiments/algorithms/ablation.yaml")
#pgf.benchmark("experiments/fairness/test.yaml", experiments, delim="&", endl="\\\\\n")

pgf.benchmark("https://raw.githubusercontent.com/maniospas/pygrank-f/main/experiments/fairness/test.yaml",
              "experiments/algorithms/fairppr.yaml",
              update=False,
              delim="&", endl="\\\\\n")
