import sys
from datetime import datetime
sys.stderr = open('err.txt', 'w')
import pygrankf as pgf
import tensorflow as tf

file = open(f'{datetime.now().strftime("results/%d-%m-%Y %H-%M-%S")}.txt', 'w')
fairobjective = pgf.loadyaml("experiments/algorithms/values.yaml")["fairobjective0"]
prule = 1


with tf.device('gpu'):
    for experiment_type in ["community"]:
        for coefficients in [pgf.PageRank(0.85), pgf.PageRank(0.9), pgf.HeatKernels(1), pgf.HeatKernels(3)]:
            algorithms = pgf.experiments("experiments/algorithms/fairsym.yaml", coefficients=coefficients,
                                         fairobjective=fairobjective, prule=prule, nsgff=None)
            pgf.benchmark(f"experiments/fairness/{experiment_type}.yaml", algorithms, total=True, file=file)
