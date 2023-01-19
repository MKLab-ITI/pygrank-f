from pyfop import *
from pygrankf.core import backend
from pygrankf.algorithms.autotune.autogf import Tunable


@lazy
@autoaspects
def culep(personalization, result, sensitive,
          culepa: list = Tunable([0, 0], [1, 1]),
          culepb: list = Tunable([-3, -3], [3, 3]),
          culept: float = Tunable(0, 1)):
    err = backend.abs(personalization-result)
    sensitive_branch = culepa[0]*backend.exp(-culepb[0]*err) + (1-culepa[0])*backend.exp(culepb[0]*err)
    non_sensitive_branch = culepa[1]*backend.exp(-culepb[1]*err) + (1-culepa[1])*backend.exp(culepb[1]*err)
    ret = sensitive*sensitive_branch + (1-sensitive)*non_sensitive_branch
    return ret * culept + personalization*(1-culept)


def neural(*inputs, layers=3, parameter_range=(-100, 100)):
    input_dims = len(inputs)
    shapes = [(input_dims, input_dims+2)] + [(input_dims+2, input_dims+2) for _ in range(layers-2)] + [(input_dims+2, 1)]
    num_params = sum([shape[0]*shape[1]+shape[1] for shape in shapes])
    first_params = list()
    for shape in shapes:
        for i in range(shape[1]):
            first_params.append(0)
        for i in range(shape[0]):
            for j in range(shape[1]):
                first_params.append(6**0.5/shape[1])#(random())*12**0.5/shape[1])
    first_params.reverse()
    tunable = Tunable([parameter_range[0]]*num_params, [parameter_range[1]]*num_params, first_params)

    @lazy_no_cache
    @autoaspects
    def neural(*args, neuralparams=tunable):
        neuralparams = neuralparams.copy()
        for shape in shapes:
            # start from biases
            next = [neuralparams.pop() for _ in range(shape[1])]
            # add dense transformation
            for i in range(shape[0]):
                for j in range(shape[1]):
                    next[j] = args[i]*neuralparams.pop() + next[j]  # potential tensor vars last to get the overriden GraphSignal operation
            # relu
            for j in range(shape[1]):
                next[j] = (next[j] + backend.abs(next[j]))/2
                #next[j].np[next[j].np < 0] = 0
            args = next
        return args[0]
    return neural(*inputs)
