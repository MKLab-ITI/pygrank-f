import sys
from datetime import datetime
sys.stderr = open('err.txt', 'w')
import pygrankf as pgf
import tensorflow as tf

file = open(f'{datetime.now().strftime("results/ablation%d-%m-%Y %H-%M-%S")}.txt', 'w')
fairobjective = pgf.loadyaml("experiments/algorithms/values.yaml")["fairobjective0"]
prule = 1

with tf.device('cpu'):
    for experiment_type in ["test"]:
        for coefficients in [pgf.PageRank(0.85), pgf.PageRank(0.9), pgf.HeatKernels(1), pgf.HeatKernels(3)]:
            algorithms = pgf.experiments("experiments/algorithms/ablation.yaml",
                                         coefficients=coefficients,
                                         appnpcoefficients=pgf.PageRank(0.9),
                                         fairobjective=fairobjective, prule=prule)
            pgf.benchmark(f"experiments/fairness/{experiment_type}.yaml", algorithms, total=True, file=file)
