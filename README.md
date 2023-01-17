# pygrank-f

This is an experimental 
[forward-oriented programming](https://github.com/maniospas/pyfop) 
variation of [pygrank](https://github.com/MKLab-ITI/pygrank);
a package for node ranking in large graphs.
`pygrank-f` provides a simpler interface on top of the
same core, but does not support all main library 
capabilities. New features successfully developed in this 
repository will be eventually fleshed out in the
interface of the source package.

**Development:** Emmanouil (Manios) Krasanakis

:cd: Large graphs<br>
:rocket: Fast processing<br>
:cookie: Easy to use<br>
:jigsaw: Modular interface<br>
:dna: Organic backpropagation<br>

## Quickstart
Graph filters are called via `pygrankf.steps` interface
that sequences the consecutive operations used to process the
first step's data. Tuning should always be the last step
and is applied on any arguments declared to be `Tunable`.

```python
import pygrankf as pgf

communities = pgf.load("citeseer")
for name, community in communities.items():
    exclude, test = community.split(0.5)
    train, validation = exclude.split(0.9)
    # define a sequence of steps
    result = pgf.steps(
        train,
        pgf.filter,
        pgf.tune(validation=validation, metric=pgf.mabs, exclude=train)
    ).call()
    pgf.print(name, pgf.auc(test, result, exclude=exclude))

```

Instead of having tunable graph filter parameters, you
can also set specific ones, alongside any other arguments
you might want to pass to your filter, such as those 
controlling its convergence:

```
.call(parameters=pgf.PageRank(alpha=0.85), errtype=pgf.mabs, tol=1.E-9)
```

In this case, the tuning will warn you that there is nothing
to be tuned and you can safely remove it from steps. Other filters 
can be declared by setting the appropriate parameters. You
can see all parameters you can set by calling `.get_input_context()`
instead of `.call()`.

You might want to define parameterized data preprocessing 
to be tuned on some objective, such as making node ranking
fairness-aware. It suffices for this definition to
produce the first step like this:

```python
original = pgf.pagerank(train).call(spectrum="symmetric")  # power method implementation of pagerank 
fair_result = pgf.steps(
    pgf.neural(train, original, sensitive),
    pgf.pagerank,
    pgf.tune(optimizer=pgf.tfsgd,
             metric=lambda *args: min(pgf.prule(args[1], sensitive, exclude), 1)-pgf.l1(args[1], original, exclude))
).call(spectrum="symmetric")  # theoretical results only for symmetric spectrums
```

The above snippet sets up a neural graph
filter prior generation scheme, which takes any number
of graph signals to account for - typically those
used to define the `metric` objective. 
The optimizer this time is 
`pgf.tfsgd`, which the package provides as 
an alternative to its own default blackbox optimization.
This requires comparatively few algorithm reruns
for convex objectives via tensorflow's autograd.
