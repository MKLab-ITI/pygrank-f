import pygrankf as pgf


#experiments = pgf.experiments("experiments/algorithms/ablation.yaml")
#pgf.benchmark("experiments/fairness/test.yaml", experiments, delim="&", endl="\\\\\n")

pgf.benchmark("experiments/fairness/diffusion.yaml",
              "experiments/algorithms/fairppr.yaml",
              update=False,
              delim="&", endl="\\\\\n")
