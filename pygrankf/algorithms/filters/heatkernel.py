from typing import Iterable


class HeatKernels(Iterable[float]):
    def __init__(self, k=5):
        self.k = k

    def __iter__(self):
        class Iterator:
            def __init__(self, k):
                self.k = k
                self.i = 1
                self.factorial = 1
                self.power = 1

            def __next__(self):
                ret = self.power / self.factorial
                self.i += 1
                self.factorial *= self.i
                self.power *= self.k
                return ret

        return Iterator(self.k)
