def steps(*args):
    result = args[0]
    for arg in args[1:]:
        result = arg(result)
    return result


def step(*args, **kwargs):
    def method(arg):
        return args[0](arg, *args[1:], **kwargs)
    return method
