def steps(*args):
    result = args[0]
    for arg in args[1:]:
        result = arg(result)
    return result
