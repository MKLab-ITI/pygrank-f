import sys

_log_prefix = ""


def prefix(text=""):
    global _log_prefix
    _log_prefix = text


def log(text=""):
    sys.stdout.write("\r" + _log_prefix + text.ljust(60))
    if len(text) == 0:
        sys.stdout.write("\r")
    sys.stdout.flush()
