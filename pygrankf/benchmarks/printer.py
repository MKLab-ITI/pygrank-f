import sys
from pyfop.execution import PendingCall


def print(*args, delim=" ", endl="\n"):
    args = [arg() if isinstance(arg, PendingCall) else arg for arg in args]
    ret = delim.join(f"{arg:.3f}".ljust(15) if isinstance(arg, float) else (str(arg).ljust(15) if not isinstance(arg, str) or len(arg) != 1 else arg) for arg in args)
    sys.stdout.write(ret+endl)
    sys.stdout.flush()
