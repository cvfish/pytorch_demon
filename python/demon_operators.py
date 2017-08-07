"""
depth to flow and flow to depth layers

"""

import numpy as np

import torch
from torch.nn.modules.module import Module

from torch.autograd import Variable

from operators import matrix_inv
from operators import axis_angle_to_rotation_matrix
from operators import perspective_projection
from operators import sample_from_images
from operators import triangulate_flow

class DepthToFlow(Module):

    def __init__(self, normalized_K = True, normalized_flow = True, inverse_depth = True):

        super(DepthToFlow, self).__init__()

        self.normalized_K = normalized_K
        self.normalized_flow = normalized_flow
        self.inverse_depth = inverse_depth

    def forward(self, K1, K2, depth, rot, trans):

        """
        :param K1: intrinsics of 1st image, 3x3
        :param K2: intrinsics of 2nd image, 3x3
        :param depth: depth map of first image, 1 x height x width
        :param rot: rotation from first to second image, 3
        :param trans: translation from first to second, 3
        :return: normalized flow from 1st image to 2nd image, 2 x height x width
        """

        height, width = depth.size()
        num = height * width

        if(self.normalized_K):
            tmp = Variable(torch.Tensor([[width, 0, 0],
                                         [0, height,0],
                                         [0,     0, 1]]))
            K1 = torch.mm(tmp, K1)
            K2 = torch.mm(tmp, K2)

        invK1 = matrix_inv(K1)
        rotation = axis_angle_to_rotation_matrix(rot)

        inv_width = 1.0 / width
        inv_height = 1.0 / height

        uu = torch.arange(0, width).expand(height, width) + 0.5
        vv = torch.arange(0,height).expand(width, height).transpose(1,0) + 0.5

        points_uvn = torch.cat([uu.contiguous().view(1, num),
                                vv.contiguous().view(1, num),
                                torch.ones(1, num)], 0)

        if(self.inverse_depth):
            points = (1.0 / depth).view(1, height * width) * \
                     torch.mm(invK1, Variable(points_uvn, requires_grad=False))
        else:
            points = depth.resize(1, height * width) * \
                     torch.mm(invK1, Variable(points_uvn, requires_grad=False))

        points2 = torch.mm(rotation, points) + trans.view(3, 1)

        points2_uv = perspective_projection(points2, K2)

        flow = points2_uv - Variable(points_uvn[0:2], requires_grad=False)

        if(self.normalized_flow):
            normalized_flow =  torch.cat([flow[0].view(height, width)*inv_width,
                                          flow[1].view(height, width)*inv_height], 0)
            return normalized_flow.view(2, height, width)
        else:
            unnormalized_flow = torch.cat([flow[0].view(height, width),
                                           flow[1].view(height, width)], 0)
            return unnormalized_flow.view(2, height, width)

class FlowToDepth(Module):

    def __init__(self, normalized_K = False, normalized_flow = True, inverse_depth = True):

        super(FlowToDepth, self).__init__()

        self.normalized_K = normalized_K
        self.normalized_flow = normalized_flow
        self.inverse_depth = inverse_depth

    def forward(self, K1, K2, flow, rot, trans):

        """
        :param K1: intrinsics of 1st image, 3x3
        :param K2: intrinsics of 2nd image, 3x3
        :param flow: flow of first image, 2 x height x width
        :param rot: rotation from first to second image, 3
        :param trans: translation from first to second, 3
        :return: depth/inv_depth of first image, 1 x height x width
        """

        rotation =  axis_angle_to_rotation_matrix(rot)
        RT = torch.cat((rotation, trans.resize(3,1)), 1)

        height, width = flow.size(1), flow.size(2)

        if(self.normalized_K):
            tmp = Variable(torch.Tensor([[width, 0, 0],
                                         [0, height,0],
                                         [0,     0, 1]]))
            K1 = torch.mm(tmp, K1)
            K2 = torch.mm(tmp, K2)

        if(self.normalized_flow):
            unnormalized_flow = torch.cat([flow[0]*width, flow[1]*height], 0).view(2, height, width)
        else:
            unnormalized_flow = flow

        I0 = Variable( torch.cat((torch.eye(3), torch.zeros(3,1)), 1))
        P1 = torch.mm( K1, I0)
        P2 = torch.mm( K2, RT)

        depth = triangulate_flow(P1, P2, unnormalized_flow)

        if(self.inverse_depth):
            return 1.0 / depth.contiguous().view(1, height, width)
        else:
            return depth.contiguous().view(1, height, width)

class WarpImage(Module):

    def forward(self, image, flow, normalized_flow = True, border_mode = 'value', border_value = 0):
        """
        :param image: input image, channels x height x width
        :param flow:  flow image, 2 x height x width
        :param normalized_flow:  whether flow is normalized or not, True or False
        :param border_mode:  border mode, 'clamp' or 'value'
        :param border_value:  border value, the value used outside the image borders.
        :return: warped image, channels x height x width
        """

        C, H, W = image.size()

        grid_v, grid_u = np.mgrid[0:H, 0:W]
        grid_v = Variable(torch.Tensor(grid_v))
        grid_u = Variable(torch.Tensor(grid_u))

        if(not normalized_flow):
            new_u = grid_u + flow[0]
            new_v = grid_v + flow[1]
        else:
            new_u = grid_u + flow[0] * W
            new_v = grid_v + flow[1] * H

        proj = torch.cat([new_u.resize(1, H * W),
                          new_v.resize(1, H * W)], 0)

        warped_image_vec = sample_from_images(proj, image,
                                              border_mode = border_mode,
                                              border_value = border_value)

        warped_image = warped_image_vec.resize(C, H, W)

        return warped_image

"""
batch depth to flow
"""
class DepthToFlowLayer(Module):

    def __init__(self, normalized_K = False,  normalized_flow = True, inverse_depth = True):

        super(DepthToFlowLayer, self).__init__()

        self.depth_to_flow = DepthToFlow(normalized_K = normalized_K,
                                         normalized_flow = normalized_flow,
                                         inverse_depth = inverse_depth)

    def forward(self, K1, K2, depth, rot, trans, shared_K = True):

        """
        :param K1:  3x3 if shared_K is True, otherwise K1 is nx3x3
        :param K2:  3x3 if shared_K is True, otherwise K2 is nx3x3
        :param depth: n x 1 x h x w
        :param rot:   n x 3
        :param trans: n x3
        :param shared_K: if True, we share intrinsics for the depth images of the whole batch
        :return: n x 2 x h x w
        """

        # depths = depth.chunk(depth.size(0), 0)
        # batch_size = len(depths)

        batch_size = depth.size(0)

        flows = ()

        for i in range(batch_size):

            if(shared_K):
                flow = self.depth_to_flow( K1, K2, depth[i][0], rot[i], trans[i])
            else:
                flow = self.depth_to_flow(K1[i], K2[i], depth[i][0], rot[i], trans[i])

            flows += (flow,)

        flow = torch.stack(flows, 0)

        return flow

"""
batch flow to depth
"""
class FlowToDepthLayer(Module):

    def __init__(self, normalized_K=False, normalized_flow=True, inverse_depth=True):

        super(FlowToDepthLayer, self).__init__()

        self.flow_to_depth = FlowToDepth(normalized_K=normalized_K,
                                         normalized_flow=normalized_flow,
                                         inverse_depth=inverse_depth)

    def forward(self, K1, K2, flow, rot, trans, shared_K = True):
        """
        :param K1:  3x3 if shared_K is True, otherwise K1 is nx3x3
        :param K2:  3x3 if shared_K is True, otherwise K2 is nx3x3
        :param flow: n x 2 x h x w
        :param rot:   n x 3
        :param trans: n x 3
        :param shared_K: if True, we share intrinsics for the depth images of the whole batch
        :return: depth, n x 1 x h x w
        """

        flows = flow.chunk(flow.size(0), 0)
        batch_size = len(flows)

        depths = ()

        for i in range(batch_size):

            if(shared_K):
                depth = self.flow_to_depth(K1, K2, flows[i][0], rot[i], trans[i])
            else:
                depth = self.flow_to_depth(K1[i], K2[i], flows[i][0], rot[i], trans[i])

            depths += (depth,)

        depth = torch.stack(depths, 0)

        return depth

"""
warpping batch images
"""
class WarpImageLayer(Module):

    def __init__(self):

        super(WarpImageLayer, self).__init__()

        self.warp_image = WarpImage()

    def forward(self, image, flow, normalized_flow = True, border_mode = 'value', border_value = 0):

        """
        :param image: image batch, n x C x H x W
        :param flow:  flow batch, n X 2 X H X W
        :param normalized_flow: True or False
        :param border_mode: 'clamp' or 'border value'
        :param border_value: the values for out of image filling
        :return: warped batch images, n x C x H x W
        """

        images = image.chunk(image.size(0), 0)

        batch_size = len(images)
        warped_images = ()

        for i in range(batch_size):

            warped_image = self.warp_image(images[i][0],
                                           flow[i],
                                           normalized_flow = normalized_flow,
                                           border_mode = border_mode,
                                           border_value = border_value)

            warped_images += (warped_image,)

        warped_image = torch.stack(warped_images, 0)

        return warped_image











