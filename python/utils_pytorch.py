

import numpy as np

import torch
from torch.autograd import Variable

import numdifftools as nd

"""
gradient checker
"""
def gradient_check(op, *args, **kwargs):

    """
    examples
    gradient_check(rigid_transformation, rot, trans, vertices, id_list=[2])
    """

    if( not 'id_list' in kwargs.keys() ):
        kwargs.update({"id_list":[0]})

    id_list = kwargs.get("id_list", [0])

    for i in id_list:

        if(not isinstance(args[i], Variable)):
            raise Exception("input {:g} is not a variable".format(i))

        if(isinstance(args[i], Variable) and not args[i].requires_grad):
            raise Exception("input {:g} doesn't require gradient".format(i))

        nelems = args[i].numel()

        """ numerical gradient """

        wrapper, p = numdiff_wrapper(op, args, kwargs, i)
        jacobian_numerical = numdiff_unified(wrapper, p)

        """ analytic gradient """

        jacobian_analytic = []

        if(len(kwargs.keys()) > 1):
            """function has dictionary inputs"""
            f = op(*args, **kwargs)
        else:
            f = op(*args)

        output_nelems = f.data.numel()

        for k in range(output_nelems):

            output_grad = torch.zeros(f.data.size())
            output_grad.view(output_nelems, 1)[k] = 1

            f.backward(output_grad, retain_variables=True)

            jacobian_analytic.append( np.copy( args[i].grad.data.view( nelems ).numpy() ) )

            for params_i in args:
                if(isinstance(params_i, torch.autograd.Variable) and params_i.requires_grad):
                    params_i.grad.data.zero_()

        jacobian_analytic = np.asarray(jacobian_analytic)

        """
        compare jacobian_analytic with jacobian_numerical
        """

        if( np.allclose(jacobian_analytic, jacobian_numerical) ):

            print "gradient is correct"

        else:

            rel_error = np.linalg.norm( jacobian_analytic - jacobian_numerical ) / \
                        np.maximum( np.linalg.norm( jacobian_analytic ), np.linalg.norm( jacobian_numerical) )

            print 'analytic jacobian :'
            print jacobian_analytic

            print 'numerical jacobian :'
            print jacobian_numerical

            print 'jacobian difference :'
            print jacobian_analytic - jacobian_numerical

            print 'relative error:'
            print rel_error

def numdiff_wrapper(func, params, keywords, i):

    """
    :param func: computational graph from pytorch
    :param params: variables of the computational graph
    :param i: the argument we want to test
    :return: the corresponding python function and evaluation point
    """

    shape = params[i].data.numpy().shape
    p = params[i].data.numpy().reshape(-1)

    def wrapper_func(input):

        """
        check if input is vector
        """
        assert len(input.shape) == 1

        params[i].data = torch.Tensor(input.reshape((shape)))

        if(len(keywords.keys()) > 1):
            outputVar = func(*params, **keywords)
        else:
            outputVar = func(*params)

        output = outputVar.data.numpy()

        return output

    return wrapper_func, p

def numdiff_unified(func, input):

    result = func(input)

    Jfunc = nd.Jacobian(func, order=10)
    J = Jfunc(input)

    if( len(input.shape) == 1 and len(result.shape) ==1 ):
        return J
    elif( len(input.shape) == 1 and len(result.shape) == 2 ):
        return J.transpose((1, 0, 2)).reshape(-1, J.shape[2])
    else:
        print 'dimension not supported for numdiff, ' \
              'input has dim{:g} and output has dim{:g}'.format(len(input.shape()), len(result.shape()))

def wrapper(func, *params, **keywords):

    """
    func is pytorch computational graph
    check each param to see if it is pytorch Variable and need grad
    """

    num = len(params)

    def from_vars_to_x(params):

        flags = [ (isinstance(params[i], Variable) and params[i].requires_grad) for i in range(num) ]

        x = np.array([]).reshape(-1,1)
        for i in range(num):
            if(flags[i]):
                x = np.concatenate((x, params[i].data.numpy().reshape((-1,1)) ))

        x = x.reshape(-1)

        return x, flags

    x0, flags = from_vars_to_x(params)

    def put_x_in_vars(x):

        pos = 0
        for i in range(num):

            if(flags[i]):

                xi = x[pos : params[i].data.numpy().size + pos]
                params[i].data =  torch.Tensor( xi.reshape(( params[i].data.numpy().shape ) ) )

                pos = pos + xi.size

        return params

    def wrapper_func(x):

        assert len(x.shape) == 1

        put_x_in_vars(x)

        if(len(keywords.keys()) >= 1):
            f = func(*params, **keywords)
        else:
            f = func(*params)

        output = f.data.numpy()

        return output

    def wrapper_func_jac(x):

        assert len(x.shape) == 1

        put_x_in_vars(x)

        """ analytic gradient """
        jacobian_analytic = []

        if(len(keywords.keys()) >= 1):
            """function has dictionary inputs"""
            f = func(*params, **keywords)
        else:
            f = func(*params)

        output_nelems = f.data.numel()

        for k in range(output_nelems):

            output_grad = torch.zeros(f.data.size())
            output_grad.view(output_nelems, 1)[k] = 1

            f.backward(output_grad, retain_variables=True)

            jacobian_analytic_i = []

            for i in range(num):
                if(flags[i]):
                    nelems = params[i].numel()
                    jacobian_analytic_i.append( np.copy( params[i].grad.data.view( nelems).numpy() ) )

            jacobian_analytic.append( np.asarray(jacobian_analytic_i).reshape(-1) )

            for params_i in params:
                if(isinstance(params_i, torch.autograd.Variable) and params_i.requires_grad):
                    params_i.grad.data.zero_()

        jacobian_analytic = np.asarray(jacobian_analytic)

        return jacobian_analytic

    return dict(x0 = x0,
                x2vars = put_x_in_vars,
                func = wrapper_func,
                jac = wrapper_func_jac)
