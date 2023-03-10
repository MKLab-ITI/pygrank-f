from pygrankf.core import utils
from random import random


def __add(weights, index, increment, max_val, min_val, coarse=0):
    """
    Adds to a value to a specific index in a list of weights while thresholding the outcome to a minimum and maximum value.
    Creates new list of weights holding the result without altering the original.

    Args:
        weights: The list of weights.
        index: The element to add to.
        increment: The value to add.
        max_val: The maximum threshold.
        min_val: The mimimum threshold.
    """
    weights = [weight for weight in weights]
    weights[index] = min(max_val, max(min_val, weights[index] + increment))
    if coarse != 0:
        weights[index] = round(weights[index] / coarse) * coarse
    return weights


def nonconvex(
    loss,
    max_vals,
    min_vals,
    starting_parameters,
    deviation_tol: float = 1.0e-9,
    divide_range: float = 2,
    partitions: int = 5,
    parameter_tol: float = float("inf"),
    depth: int = 1,
    coarse: float = 0,
    shrink_strategy: str = "divide",
    partition_strategy: str = "split",
    randomize_search: bool = False,
    verbose: bool = True,
):
    """
    Implements a coordinate descent algorithm for optimizing the argument vector of the given loss function.
    A simplified version for default parameters has been published in [krasanakis2022autogf].
    Arguments:
        loss: The loss function. Could be an expression of the form `lambda p: f(p)' where f takes a list as an argument.
        max_vals: Optional. The maximum value for each parameter to search for. Helps determine the number of parameters.
            Default is a list of ones for one parameter.
        min_vals: Optional. The minimum value for each paramter to search for. If None (default) it becomes a list of
            zeros and equal length to max_vals.
        deviation_tol: Optional. The numerical tolerance of the loss to optimize to. Default is 1.E-8.
        divide_range: Optional. Value greater than 1 with which to divide the range at each iteration. Default is 1.01,
            which guarantees convergence even for difficult-to-optimize functions, but values such as 1.1, 1.2 or 2 may
            also be used for much faster, albeit a little coarser, convergence. If the *shrink_strategy* argument
            is set to "shrinking" instead, the range is scaled proportionally to
            *iteration<sup>divide_range</sup>/log(iteration)* per block coordinate descent.
        partitions: Optional. In how many pieces to break the search space on each iteration. Default is 5.
        parameter_tol: Optional. The numerical tolerance of parameter values to optimize to. **Both** this and
            deviation_tol need to be met. Default is infinity.
        depth: Optional. Declares the number of times to re-perform the optimization given the previous found solution.
            Default is 1, which only runs the optimization once. Larger depth values can help offset coarseness
            introduced by divide_range.
        coarse: Optional. Optional. Snaps solution to this precision. If 0 (default) then this behavior is ignored.
        shrink_strategy: Optional. The shrinking strategy towards convergence. If "divide" (default), then
            the search range is divided by the argument *divide_range*, but if "shrinking" then it is
            scaled based on block coordinate descent.
        partition_strategy: Optional. Strategy with which to traverse partitions. If "split" (default), then
            the partition is split to *partitions* parts. If "step", then the *partitions* argument is used as a fixed
            step and however many splits are needed to achieve this are performed. This last strategy helps
            force block coordinate descent traverse a finite set of values, as long as it holds that
            **coarse==partitions**.
        randomize_search: Optional. If True (default), then a random parameter is updated each time instead of moving
            though them in a cyclic order.
        starting_parameters: Optional. An estimation of parameters to start optimization from. The algorithm tries to center
            solution search around these - hence the usefulness of *depth* as an iterative scheme. If None (default),
            the center of the search range (max_vals+min_vals)/2 is used as a starting estimation.
        verbose: Options. If True, optimization outputs its intermediate steps. Default is False.
    Example:
        >>> import pygrankf as pg
        >>> p = pg.nonconvex(loss=lambda p: (1.5-p[0]+p[0]*p[1])**2+(2.25-p[0]+p[0]*p[1]**2)**2+(2.625-p[0]+p[0]*p[1]**3)**2, max_vals=[4.5, 4.5], min_vals=[-4.5, -4.5])
        >>> # desired optimization point for the Beale function of this example is [3, 0.5]
        >>> print(p)
        [3.000000052836577, 0.5000000141895036]
    """
    for min_val, max_val in zip(min_vals, max_vals):
        assert min_val <= max_val
    if str(divide_range) != "shrinking":
        assert divide_range > 1
    if starting_parameters is None:
        weights = [
            (max_val + min_val) / 2 for min_val, max_val in zip(min_vals, max_vals)
        ]
    else:
        weights = starting_parameters
    range_search = [
        (max_val - min_val) / 2 for min_val, max_val in zip(min_vals, max_vals)
    ]
    curr_variable = 0
    iter = 0
    range_deviations = [float("inf")] * len(max_vals)
    best_weights = weights
    best_loss = float("inf")
    evals = 0
    while True:
        if randomize_search:
            curr_variable = int(random() * len(weights))
        if max(range_search) == 0:
            break
        assert (
            max(range_search) != 0
        ), "Something went wrong and took too many iterations for optimizer to run (check for nans)"
        if shrink_strategy == "shrinking":
            range_search[curr_variable] = (
                max_vals[curr_variable] - min_vals[curr_variable]
            ) / ((iter + 1) ** divide_range * log(iter + 2))
        elif shrink_strategy == "divide":
            range_search[curr_variable] /= divide_range
        else:
            raise Exception(
                "Invalid shrink strategy: either shrinking or divide expected"
            )
        if range_search[curr_variable] == 0:
            range_deviations[curr_variable] = 0
            curr_variable += 1
            if curr_variable >= len(max_vals):
                curr_variable -= len(max_vals)
            continue
        if partition_strategy == "split":
            candidate_weights = [
                __add(
                    weights,
                    curr_variable,
                    range_search[curr_variable] * (part * 2.0 / (partitions - 1) - 1),
                    max_vals[curr_variable],
                    min_vals[curr_variable],
                    coarse=coarse,
                )
                for part in range(partitions)
            ]
        elif partition_strategy == "step":
            candidate_weights = [
                __add(
                    weights,
                    curr_variable,
                    part * partitions,
                    max_vals[curr_variable],
                    min_vals[curr_variable],
                    coarse=coarse,
                )
                for part in range(
                    -int(range_search[curr_variable] / partitions),
                    1 + int(range_search[curr_variable] / partitions),
                )
            ]
        else:
            raise Exception("Invalid partition strategy: either split or step expected")
        loss_pairs = [(w, loss(w)) for w in candidate_weights if w is not None]
        evals += len(loss_pairs)
        weights, weights_loss = min(loss_pairs, key=lambda pair: pair[1])
        prev_best_loss = best_loss
        best_loss = weights_loss
        best_weights = weights
        #range_deviations[curr_variable] = abs(prev_best_loss - best_loss)
        range_deviations[curr_variable] = max(v[1] for v in loss_pairs) - min(v[1] for v in loss_pairs)
        if verbose:
            utils.log(
                f"Tuning evaluations {evals} loss {best_loss:.8f} +- {max(range_deviations):.8f}"
            )

        if (
            max(range_deviations) <= deviation_tol
            and max(range_search) <= parameter_tol
        ):
            break
        # move to next var
        iter += 1
        curr_variable += 1
        if curr_variable >= len(max_vals):
            curr_variable -= len(max_vals)
    weights = best_weights
    if verbose:
        utils.log()
    if depth > 1:
        return nonconvex(
            loss,
            max_vals,
            min_vals,
            weights,
            deviation_tol,
            divide_range,
            partitions,
            parameter_tol,
            depth - 1,
            coarse,
            shrink_strategy,
            partition_strategy,
            randomize_search,
            verbose,
        )
    return weights
