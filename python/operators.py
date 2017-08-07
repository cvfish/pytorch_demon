"""
useful geometric operators for demon net

axis_angle_to_rotation_matrix

perspective_projection

sample_from_images

"""

import numpy as np

import torch
from torch.autograd import Variable
from torch.autograd import Function
from torch.nn.modules.module import Module

"""
convert from axis-angle representation to rotation matrix
http://www.ethaneade.com/latex2html/lie_groups/node37.html

w in so3 to rotation matrix:

R = exp(w_x) = I + (sin(\theta) / \theta)w_x + ((1-cos(\theta))\theta^2) w_x^2

derivative dR_dwi:

      --- w_i [w]x + [w x (I - R)e_i]x
	  ----------------------------------- R
      --- 	 ||w||^{2}

w: 3
R: 3x3

"""

class AxisAngleToRotationMatrix(Function):

    def forward(self, w):

        R = self.forward_core(w)

        self.save_for_backward(w, R)

        return R

    @staticmethod
    def forward_core(w):

        theta = torch.norm(w)

        if (theta > 0):

            wx = torch.Tensor([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            R = torch.eye(3) + np.sin(theta) / theta * wx + ((1 - np.cos(theta)) / theta ** 2) * wx.mm(wx)

        else:

            R = torch.Tensor([[1, -w[2], w[1]], [w[2], 1, -w[0]], [-w[1], w[0], 1]])

        return R

    def backward(self, grad_output):

        w, R = self.saved_tensors

        grad_w = self.backward_core(w, R, grad_output)

        return grad_w

    @staticmethod
    def backward_core(w, R, grad_output):

        grad_w = torch.zeros(3)

        theta = torch.norm(w)

        if (theta > 0):

            wx = torch.Tensor([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

            for i in range(3):
                ei = torch.zeros(3, 1);
                ei[i] = 1
                temp = wx.mm((torch.eye(3) - R)).mm(ei)

                dR_dwi = (w[i] * wx + torch.Tensor([[0, -temp[2][0], temp[1][0]],
                                                    [temp[2][0], 0, -temp[0][0]],
                                                    [-temp[1][0], temp[0][0], 0]])).mm(R) / theta ** 2

                grad_w[i] = (grad_output * dR_dwi).sum()

        else:

            grad_w[0] = grad_output[2][1] - grad_output[1][2]
            grad_w[1] = grad_output[0][2] - grad_output[2][0]
            grad_w[2] = grad_output[1][0] - grad_output[0][1]

        return grad_w

def axis_angle_to_rotation_matrix(w):

    return AxisAngleToRotationMatrix()(w)

""" matrix inversion """
class MatrixInv(Function):

    """even give variable as input, in forward function, A will be torch.Tensor """
    def forward(self, A):

        C = torch.inverse(A)
        self.save_for_backward(C)

        return C

    def backward(self, grad_output):

        C = self.saved_tensors[0]

        return -C.transpose(1,0).mm( grad_output ).mm( C.transpose(1,0) )

def matrix_inv(A):

    return MatrixInv()(A)

"""
my svd implementation which supports backprop
reference "https://arxiv.org/pdf/1509.07838.pdf", page 7, proposition 1
"""
class MySVD(Function):

    def forward(self, X):

        """
        :param X: m x n, m >= n
        :return: U(m x m), S(n x n) V(n x n)
        """

        if(X.size(0) >= X.size(1)):

            U, S, V = torch.svd(X, some=False)

            """check the singular values to make sure they are different"""
            n = S.size(0)
            diff = S.view(n, 1) - S.view(1,n)
            if(torch.sum( diff == 0 ) > n):
                print "there are same singular values"

            S = torch.diag(S)

            self.save_for_backward(U, S, V)
            return U, S, V

            # """for testing backward"""
            # USV = torch.cat((U, S, V), 1)
            # self.save_for_backward(USV)
            # return USV

        else:

            print "we need X(m x n) has size m >= n"

    # """for testing backward"""
    # def backward(self, grad_output):
    #
    #     USV = self.saved_tensors[0]
    #     m = USV.size(0)
    #     n = m
    #
    #     U = USV[:, 0:m]
    #     S = USV[:, m:2*m]
    #     V = USV[:, 2*m:3*m]
    #
    #     pL_pU = grad_output[:, 0:m]
    #     pL_pS = grad_output[:, m:2*m]
    #     pL_pV = grad_output[:, 2*m:3*m]

    def backward(self, *grad_output):

        U, S, V = self.saved_tensors

        m = U.size(0)
        n = V.size(0)

        pL_pU = grad_output[0]
        pL_pS = grad_output[1]
        pL_pV = grad_output[2]

        S_inv = torch.inverse(S)

        if(m > n):
            D = torch.mm( pL_pU[:, 0:n], S_inv ) - \
                torch.mm( U[:, n:], torch.mm( pL_pU[:, n:].t(), torch.mm( U[:, 0:n], S_inv ) ) )
        else:
            D = torch.mm(pL_pU[:, 0:n], S_inv)

        S_full = torch.cat((S, torch.zeros(m-n, n)), 0)
        pL_pS_full = torch.cat((pL_pS, torch.zeros(m-n, n)), 0)

        pL_pX_1 = torch.mm( D, V.t() )

        temp = torch.diag(torch.diag(pL_pS_full - torch.mm(U.t(), D)))
        pL_pX_2 = torch.mm( U,  torch.mm( temp, V.t() ))

        S2 = S*S
        K = torch.diag( S2 ).view(n,1) - torch.diag( S2 ).view(1, n)
        K = torch.pow(K, -1)
        K[torch.max(K) == K] = 0
        temp = torch.mm(V.t(), (pL_pV - torch.mm(V, torch.mm(D.t(), torch.mm( U, S_full )) ) ))
        temp2 = K.t() * temp
        temp_sym = 0.5 * (temp2 + temp2.t())
        pL_pX_3 = 2 * torch.mm(U, torch.mm(S_full, torch.mm(temp_sym, V.t())) )

        return pL_pX_1 + pL_pX_2 + pL_pX_3

def my_svd(A):

    return MySVD()(A)

"""
svd tailored for triangulation.
In triangulation, we only care about the last column of V
"""
class MySVDTrig(Function):

    def forward(self, X):

        """
        :param X: m x n, m >= n
        :return: U(m x m), S(n x n) V(n x n)
        """

        if (X.size(0) >= X.size(1)):

            U, S, V = torch.svd(X, some=False)

            """check the singular values to make sure they are different"""
            n = S.size(0)
            diff = S.view(n, 1) - S.view(1, n)
            if (torch.sum(diff == 0) > n):
                print "there are same singular values"

            S = torch.diag(S)

            self.save_for_backward(X)

            return V[:,-1]

        else:

            print "we need X(m x n) has size m >= n"

    def backward(self, grad_output):

        X = self.saved_tensors[0]
        U, S, V = torch.svd(X, some=False)
        S = torch.diag(S)

        # U, S, V = self.saved_tensors

        m = U.size(0)
        n = V.size(0)

        pL_pV = torch.cat((torch.zeros(n, n-1), grad_output), 1)

        if(m > n):
            S_full = torch.cat((S, torch.zeros(m - n, n)), 0)
        else:
            S_full = S

        S2 = S * S
        K = torch.diag(S2).view(n, 1) - torch.diag(S2).view(1, n)
        K = torch.pow(K, -1)
        K[torch.max(K) == K] = 0
        temp = torch.mm(V.t(), pL_pV)
        temp2 = K.t() * temp
        temp_sym = 0.5 * (temp2 + temp2.t())
        pL_pX = 2 * torch.mm(U, torch.mm(S_full, torch.mm(temp_sym, V.t())))

        return pL_pX

def my_svd_trig(A):

    return MySVDTrig()(A)

""" rigid transformation """
"""
rot: 3
trans: 3
vertices: 3xn
X: 3xn
"""
def rigid_transformation(rot, trans, vertices):

    rot_mat = axis_angle_to_rotation_matrix(rot)
    X = rot_mat.mm(vertices) + trans.view(3, 1)

    return X

""" perspective projection """
"""
X: 3xn
K: 3x3
"""
def perspective_projection(X, K):

    KX = K.mm(X)
    num = X.size()[1]
    ProjX = torch.div(KX[0:2], KX[2].view(1, num))

    return ProjX

""" triangulation flow"""
"""
P1: P matrix of first image
P2: P matrix of second image
flow: optical flow from first image to second image
return: return the depth value of points based on flow and P matrices
"""
def triangulate_flow(P1, P2, flow):

    H, W = flow.size(1), flow.size(2)

    grid_v, grid_u = np.mgrid[0:H, 0:W]

    grid_v = Variable(torch.Tensor(grid_v + 0.5))
    grid_u = Variable(torch.Tensor(grid_u + 0.5))

    uv1 = torch.cat((grid_u.view(1, H*W),
                     grid_v.view(1, H*W)), 0)

    uv2 = uv1 + torch.cat((flow[0].view(1, H*W),
                           flow[1].view(1, H*W)), 0)

    Ps = torch.stack((P1, P2), 0)
    W = torch.cat((uv1, uv2), 0)

    XYZ = triangulate_points(Ps, W)

    depth = XYZ[:,2]

    return depth

""" triangulation points"""
"""
Ps: K x 3 x 4
W: 2K x P
return: triangulation results XYZ, P x 3
"""
def triangulate_points(Ps, W):

    cameras = Ps.size(0)
    points = W.size(1)

    A12 = torch.cat((Ps[:, 0, :], Ps[:, 1, :]), 0)
    A33 = torch.cat((Ps[:, 2, :], Ps[:, 2, :]), 0)

    UV = torch.cat( (W[0:cameras*2:2, :],
                     W[1:cameras*2:2, :]), 0)

    XYZ = ()

    for j in range(points):

        A = A12 - A33 * UV[:, j].contiguous().view(-1,1)

        """
        fix me, write a svd layer which supports backprop
        A = U * S * V',
        here we just need my_svd_trig, return only the last column of V
        """
        # U, S, V = my_svd(A)
        # X = V[:, -1]

        X = my_svd_trig(A)

        xyz = X[0:3] / X[3]

        XYZ += (xyz, )

    return torch.cat(XYZ, 0).view(points, 3)

"""
sampling values from C * H * W images based on 2d projections
img: C * H * W
proj: 2 * P
values: C * P
img is provided as parameters, proj as input and sampled values as output
"""

class SamplingFromImages(Module):

    def forward(self, proj, img, border_mode = 'clamp', border_value = 0):

        C, H, W = img.size()

        if border_mode == 'clamp':

            proj_u = torch.clamp(proj[0], min=0, max=W - 1)
            proj_v = torch.clamp(proj[1], min=0, max=H - 1)

            ul = proj_u.int()
            vl = proj_v.int()

            delta_u = proj_u - ul.float()
            delta_v = proj_v - vl.float()

            # uu = torch.clamp(ul + 1, min=0, max=W - 1)
            # vu = torch.clamp(vl + 1, min=0, max=H - 1)

            """To make sure this is exactly the same as DeMoN"""
            uu = torch.clamp(proj[0] + 1, min=0, max=W - 1).int()
            vu = torch.clamp(proj[1] + 1, min=0, max=H - 1).int()

            # img_vec = img.view(C, H * W)
            img_vec = img.contiguous().view(C, H * W)

            img_ll = img_vec[:, (vl * W + ul).data.long()]
            img_ul = img_vec[:, (vu * W + ul).data.long()]
            img_lu = img_vec[:, (vl * W + uu).data.long()]
            img_uu = img_vec[:, (vu * W + uu).data.long()]

            values = (1 - delta_u) * (1 - delta_v) * img_ll + \
                     (1 - delta_u) * delta_v * img_ul + \
                     delta_u * (1 - delta_v) * img_lu + \
                     delta_u * delta_v * img_uu

            return  values

        elif border_mode == 'value':

            ul = proj[0].int()
            vl = proj[1].int()

            delta_u = proj[0] - ul.float()
            delta_v = proj[1] - vl.float()

            uu = ul + 1
            vu = vl + 1

            """The up to date github version """
            # mask = (ul >= 0) * (uu > 0) * (uu < W) * (vl >= 0) * (vu > 0) * (vu < H)

            """To make sure this is exactly the same as DeMoN code (the version on my desktop)"""
            mask = (ul >= 0) * (uu < W) * (vl >= 0) * (vu < H)

            ul = torch.clamp(ul, min=0, max=W - 1)
            vl = torch.clamp(vl, min=0, max=H - 1)
            uu = torch.clamp(uu, min=0, max=W - 1)
            vu = torch.clamp(vu, min=0, max=H - 1)

            # img_vec = img.view(C, H * W)
            img_vec = img.contiguous().view(C, H * W)

            img_ll = img_vec[:, (vl * W + ul).data.long()]
            img_ul = img_vec[:, (vu * W + ul).data.long()]
            img_lu = img_vec[:, (vl * W + uu).data.long()]
            img_uu = img_vec[:, (vu * W + uu).data.long()]

            # """this will also work in the pytorch0.2"""
            # img_ll = img.contiguous()[:, vl.data.long(), ul.data.long()]
            # img_ul = img.contiguous()[:, vu.data.long(), ul.data.long()]
            # img_lu = img.contiguous()[:, vl.data.long(), uu.data.long()]
            # img_uu = img.contiguous()[:, vu.data.long(), uu.data.long()]

            values = (1 - delta_u) * (1 - delta_v) * img_ll + \
                     (1 - delta_u) * delta_v * img_ul + \
                     delta_u * (1 - delta_v) * img_lu + \
                     delta_u * delta_v * img_uu

            boundary_mask = (mask == 0)

            values = values * mask.float() + \
                     border_value * boundary_mask.float()

            return values

"""
hand coded version,
assume img is fixed
code back propagation manually
"""
class MySamplingFromImages(Function):

    def __init__(self, img):

        self.img = img

    def forward(self, proj):

        img = self.img
        C, H, W = img.size()

        proj_u = torch.clamp( proj.data[0], min=0, max=W-1)
        proj_v = torch.clamp( proj.data[1], min=0, max=H-1)

        ul = proj_u.int()
        vl = proj_v.int()

        delta_u = proj_u - ul
        delta_v = proj_v - vl

        uu = torch.clamp(ul + 1, min=0, max=W-1)
        vu = torch.clamp(vl + 1, min=0, max=H-1)

        img_vec = img.view(C, H*W)

        img_ll = img_vec[:, vl*W + ul]
        img_ul = img_vec[:, vu*W + ul]
        img_lu = img_vec[:, vl*W + uu]
        img_uu = img_vec[:, vu*W + uu]

        values = (1 - delta_u) * (1 - delta_v) * img_ll + \
                 (1 - delta_u) * delta_v * img_ul + \
                 delta_u * (1 - delta_v) * img_lu + \
                 delta_u * delta_v * img_uu

        self.save_for_backward(proj.data, img_ll, img_ul, img_lu, img_uu)

        return values

    def backward(self, grad_output):

        """
        :param grad_output: C x P
        :return: 2 x P
        """

        img = self.img
        proj, img_ll, img_ul, img_lu, img_uu = self.saved_tensors

        C, H, W = img.size()
        P = proj.size(1)

        dv_dp = torch.zeros(C, 2, P)

        grad_proj = torch.zeros(2, P)

        mask = (proj[0] > 0) * (proj[0] < W - 1) * (proj[1] > 0) * (proj[1] <  H - 1)
        ind = torch.nonzero(mask)

        ul = proj[0][mask].int()
        vl = proj[1][mask].int()

        delta_u = proj[0] - ul
        delta_v = proj[1] - vl

        dv_dp[:, 0, :][:, ind] = -(1 - delta_v) * img_ll - \
                                 delta_v * img_ul + \
                                 (1 - delta_v) * img_lu + \
                                 delta_v * img_uu

        dv_dp[:, 1, :][:, ind] = -(1 - delta_u) * img_ll + \
                                 (1 - delta_u) * img_ul - \
                                 delta_u * img_lu + \
                                 delta_u * img_uu

        grad_proj[0, :] = ( grad_output * dv_dp[:, 0, :]).sum(0)
        grad_proj[1, :] = ( grad_output * dv_dp[:, 1, :]).sum(0)

        return grad_proj

def sample_from_images(proj, img, border_mode = 'clamp', border_value = 0):

    return SamplingFromImages()(proj, img, border_mode = border_mode, border_value = border_value)
