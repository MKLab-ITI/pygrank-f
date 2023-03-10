import sys
from pyfop.execution import PendingCall


def print(*args, delim=" ", endl="\n", file=None):
    args = [arg() if isinstance(arg, PendingCall) else arg for arg in args]
    ret = delim.join(
        f"{arg:.3f}".ljust(9)
        if isinstance(arg, float)
        else (str(arg).ljust(9) if not isinstance(arg, str) or len(arg) != 1 else arg)
        for arg in args
    )
    if file is None:
        file = sys.stdout
    file.write(ret + endl)
    file.flush()
