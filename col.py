import sys
#sys.stderr = open('err.txt', 'w')
import pygrankf as pgf
import tensorflow as tf


with tf.device('gpu'):
    for experiment_type in ["community", "diffusion"]:
        for coefficients in [pgf.PageRank(0.85), pgf.PageRank(0.9)]:
            algorithms = pgf.experiments("experiments/algorithms/fairppr.yaml", coefficients=coefficients)
            pgf.benchmark(f"experiments/fairness/{experiment_type}.yaml", algorithms, total=True)

        for coefficients in [pgf.HeatKernels(1), pgf.HeatKernels(3)]:
            algorithms = pgf.experiments("experiments/algorithms/fairany.yaml", coefficients=coefficients)
            pgf.benchmark(f"experiments/fairness/{experiment_type}.yaml", algorithms, total=True)

        for coefficients in [pgf.PageRank(0.85), pgf.PageRank(0.9), pgf.HeatKernels(1), pgf.HeatKernels(3)]:
            algorithms = pgf.experiments("experiments/algorithms/fairsym.yaml", coefficients=coefficients)
            pgf.benchmark(f"experiments/fairness/{experiment_type}.yaml", algorithms, total=True)
