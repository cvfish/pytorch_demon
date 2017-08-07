"""
test file
"""

"""
gradient test, compare autograd gradient with numeric gradient for each operation
rotation, rigid transformation, projection, sampling
"""

import numpy as np

import torch
from torch.autograd import Variable

from operators import axis_angle_to_rotation_matrix
from operators import rigid_transformation
from operators import perspective_projection
from operators import sample_from_images

from operators import matrix_inv
from operators import my_svd
from operators import my_svd_trig

from utils_diff import numdiff_jacobian

from utils_pytorch import wrapper
from utils_pytorch import gradient_check

"""
rotation operation
"""

# rot = Variable( torch.Tensor( np.random.rand(3) ), requires_grad = True )
rot = Variable( torch.Tensor( [ 0.56004662,  0.97220967,  0.71514336] ), requires_grad = True )
gradient_check(axis_angle_to_rotation_matrix, rot)

func_wrapper  = wrapper(axis_angle_to_rotation_matrix, rot )
func, jac, x0 = func_wrapper['func'], func_wrapper['jac'], func_wrapper['x0']

print numdiff_jacobian(func, x0)
print jac(x0)

"""
matrix inverse
"""

print "matrix inverse"
mat = Variable( torch.randn(3,3), requires_grad = True )
gradient_check(matrix_inv, mat)

"""
svd_trig
"""

print "svd_trig"
mat = Variable( torch.randn(4,4), requires_grad = True )
gradient_check(my_svd_trig, mat)

"""
svd, need to turn on testing mode
"""
print "svd"
mat = Variable( torch.randn(4,4), requires_grad = True )
my_svd(mat)
# gradient_check(my_svd, mat)

"""rigid transformation """
trans = Variable( torch.Tensor( np.random.rand(3) ), requires_grad = True )
vertices = Variable( torch.Tensor( np.random.rand(3, 1) ) , requires_grad = False )

"""rotation"""
gradient_check(rigid_transformation, rot, trans, vertices, id_list=[0])
"""translation"""
gradient_check(rigid_transformation, rot, trans, vertices, id_list=[1])
"""vertices"""
# print "rigid transformation, vertices gradient"
# gradient_check(rigid_transformation, rot, trans, vertices, id_list=[2])

func_wrapper  = wrapper( rigid_transformation, rot, trans, vertices )
func, jac, x0 = func_wrapper['func'], func_wrapper['jac'], func_wrapper['x0']

print numdiff_jacobian(func, x0)
print jac(x0)

"""perspective projection"""
K = Variable( torch.Tensor( np.random.rand(3,3) + 1 ), requires_grad = True )
X = Variable( torch.Tensor( np.random.rand(3, 1) ), requires_grad = True )
gradient_check( perspective_projection, X, K, id_list=[0])

""" image sampling """
print "image sampling"
img = Variable(torch.from_numpy(np.random.rand(10,10,1)).float(), requires_grad = True)
proj = Variable(
    torch.Tensor(
        np.array([[np.random.uniform(1,2) for i in range(2)] for j in range(2)])),
    requires_grad = True )

gradient_check(sample_from_images, proj, img, id_list=[0])
gradient_check(sample_from_images, proj, img, id_list=[1])