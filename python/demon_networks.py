""""
pytorch implementation of DeMoN - Depth and Motion Network

this file provides the implementation of DeMoN architecture,

including BootstrapNet, IterativeNet and RefinementNet

"""


from blocks import *
import torch.nn as nn

"""
BootstrapNet, without any previous predictions
"""
class BootstrapNet(nn.Module):

    def __init__(self):

        super(BootstrapNet, self).__init__()

        self.flow_block = FlowBlock(use_prev_predictions=False)
        self.depth_motion_block = DepthMotionBlock(use_prev_depthmotion=False)


    def forward(self, image_pair, image2_2):

        flow = self.flow_block(image_pair)

        predictions = self.depth_motion_block(image_pair, image2_2, flow[:, 0:2, :, :], flow)

        return predictions

"""
IterativeNet, use previous depth and motion information
"""

class IterativeNet(nn.Module):

    def __init__(self):

        super(IterativeNet, self).__init__()

        self.flow_block = FlowBlock(use_prev_predictions=True)
        self.depth_motion_block = DepthMotionBlock(use_prev_depthmotion=True)

    def forward(self, image_pair, image2_2, intrinsics, prev_predictions):

        flow = self.flow_block(image_pair, image2_2, intrinsics, prev_predictions)

        predictions = self.depth_motion_block(image_pair, image2_2,
                                              flow[:, 0:2, :, :], flow,
                                              intrinsics=intrinsics,
                                              prev_predictions=prev_predictions)

        return predictions

"""
RefinementNet, refine depth output
"""

class RefinementNet(nn.Module):

    def __init__(self):

        super(RefinementNet, self).__init__()

        self.refinement_block = RefinementBlock()

    def forward(self, image1, depth):

        refinement = self.refinement_block(image1, depth)

        return refinement







