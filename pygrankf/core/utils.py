import sys


def log(text=""):
    sys.stdout.write("\r" + text.ljust(60))
    if len(text) == 0:
        sys.stdout.write("\r")
    sys.stdout.flush()
