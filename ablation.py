import sys
sys.stderr = open('err.txt', 'w')
import pygrankf as pgf
import tensorflow as tf


with tf.device('cpu'):
    for coefficients in [pgf.PageRank(0.85), pgf.PageRank(0.90), pgf.HeatKernels(1), pgf.HeatKernels(3)]:
        algorithms = pgf.experiments("experiments/algorithms/fairppr.yaml", coefficients=coefficients)
        pgf.benchmark("experiments/fairness/links.yaml", algorithms,
                      update=True, delim="&", endl="\\\\\n", total=True)

pgf.sweep()