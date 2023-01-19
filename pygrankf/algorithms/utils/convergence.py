from timeit import default_timer as time
from pygrankf.core import backend, BackendPrimitive
from pygrankf.measures import mabs
from pyfop import *


@lazy_no_cache
@autoaspects
class Convergence:
    """ Used to keep previous iteration and generally manage convergence of variables. Graph filters
    automatically create instances of this class by passing on appropriate parameters.

    Examples:
        >>> convergence = Convergence()
        >>> convergence.start()
        >>> var = None
        >>> while not convergence.has_converged(var):
        >>>     ...
        >>>     var = ...
    """
    def __init__(self,
                 tol: float = 1.E-6,
                 errortype: str = "iters",
                 maxiters: int = 20):
        """
        Initializes a convergence manager with a provided tolerance level, error type and number of iterations.

        Args:
            tol: Numerical tolerance to determine the stopping point (algorithms stop if the "error" between
                consecutive iterations becomes less than this number). Default is 1.E-6 but for large graphs
                1.E-9 often yields more robust convergence points. If the provided value is less than the
                numerical precision of the backend `pygrank.epsilon()` then it is snapped to that value.
            errortype: Optional. How to calculate the "error" between consecutive iterations of graph signals.
                If "iters", convergence is reached at iteration *max_iters*-1 without throwing an exception.
                Default is `pygrank.Mabs`.
            maxiters: Optional. The number of iterations algorithms can run for. If this number is exceeded,
                an exception is thrown. This could help manage computational resources. Default value is 100,
                and exceeding this value with graph filters often indicates that either graphs have large diameters
                or that algorithms of choice converge particularly slowly.
        """
        self.tol = tol
        self.error_type = errortype
        self.max_iters = maxiters
        self.iteration = 0
        self.last_ranks = None
        self._start_time = None
        self.elapsed_time = None

    def start(self, restart_timer: bool = True):
        """
        Starts the convergence manager

        Args:
            restart_timer: Optional. If True (default) timing information, such as the number of iterations and wall
                clock time measurement, is reset. Otherwise, this only ensures that the convergence manager
                performs one iteration before starting comparing values with previous ones.
        """
        if restart_timer or self._start_time is None:
            self._start_time = time()
            self.elapsed_time = None
            self.iteration = 0
        self.last_ranks = None

    def has_converged(self, new_ranks: BackendPrimitive) -> bool:
        """
        Checks whether convergence has been achieved by comparing this iteration's backend array with the
        previous iteration's.

        Args:
            new_ranks: The iteration's backend array.
        """
        self.iteration += 1
        if self.iteration >= self.max_iters:
            if self.error_type == "iters":
                self.elapsed_time = time()-self._start_time
                return True
            raise Exception("Could not converge within "+str(self.max_iters)+" iterations")
        converged = False if self.last_ranks is None else self._has_converged(self.last_ranks, new_ranks)
        self.last_ranks = new_ranks
        self.elapsed_time = time()-self._start_time
        return converged

    def _has_converged(self, prev_ranks: BackendPrimitive, ranks: BackendPrimitive) -> bool:
        if self.error_type == "iters":
            return False
        return self.error_type(prev_ranks, ranks) <= max(self.tol, backend.epsilon())

    def __str__(self):
        return str(self.iteration)+" iterations ("+str(self.elapsed_time)+" sec)"
