import numdifftools as nd

"""
only useful when input is single dimension, while output is two dimension
"""

def numdiff_jacobian(func, input, order=2):

    result = func(input)

    Jfunc = nd.Jacobian(func, order=order)
    J = Jfunc(input)

    if( len(result.shape) == 2 ):
        J = J.transpose((1, 0, 2)).reshape(-1, J.shape[2])

    return J

