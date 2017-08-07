""""
pytorch implementation of DeMoN - Depth and Motion Network

this file provides the implementation of basic blocks used in DeMoN architecture,

including FlowBlock, DepthMotBlock, RefinementBlock

"""

import torch
import torch.nn as nn

from demon_operators import WarpImageLayer
from demon_operators import DepthToFlowLayer
from demon_operators import FlowToDepthLayer

def convrelu2_block( num_inputs, num_outputs , kernel_size, stride, leaky_coef ):

    """
    :param num_inputs: number of input channels
    :param num_outputs:  number of output channels
    :param kernel_size:  kernel size
    :param stride:  stride
    :param leaky_coef:  leaky ReLU coefficients
    :return: 2x(Conv + ReLU) block
    """

    """ this block does two 1D convolutions, first on row, then on column """

    input = num_inputs; output = num_outputs
    k  = kernel_size; lc = leaky_coef

    if( not isinstance(stride, tuple)):
        s = (stride, stride)
    else:
        s = stride

    conv1_1 = nn.Conv2d( input,  output[0], (k[0], 1), padding=(k[0] // 2, 0), stride=(s[0], 1) )
    leaky_relu1_1 = nn.LeakyReLU( lc )

    conv1_2 = nn.Conv2d( output[0],  output[1], (1, k[1]), padding=(0, k[1] // 2), stride=(1, s[1]) )
    leaky_relu1_2 = nn.LeakyReLU( lc )

    return nn.Sequential(
        conv1_1,
        leaky_relu1_1,
        conv1_2,
        leaky_relu1_2
    )

def convrelu_block( num_inputs, num_outputs, kernel_size, stride, leaky_coef ):

    """
    :param num_inputs: number of input channels
    :param num_outputs:  number of output channels
    :param kernel_size:  kernel size
    :param stride:  stride
    :param leaky_coef:  leaky ReLU coefficients
    :return: (Conv + ReLU) block
    """

    """ this block does one 2D convolutions, first on row, then on column """

    input = num_inputs; output = num_outputs
    k = kernel_size; lc = leaky_coef

    if( not isinstance(stride, tuple)):
        s = (stride, stride)
    else:
        s = stride

    conv1_1 = nn.Conv2d(input, output, k, padding=(k[0] // 2, k[1] // 2), stride=s )
    leaky_relu1_1 = nn.LeakyReLU(lc)

    return nn.Sequential(
        conv1_1,
        leaky_relu1_1
    )

def predict_flow_block( num_inputs, num_outputs=4, intermediate_num_outputs=24):
    """
    :param num_inputs: number of input channels
    :param predict_confidence:  predict confidence or not
    :return: block for predicting flow
    """

    """"
    this block is --> (Conv+ReLU) --> Conv --> ,

    in the first prediction,  input is 512 x 8 x 6,
    in the second prediction, input is 128 x 64 x 48

    """

    conv1 = convrelu_block( num_inputs, intermediate_num_outputs,  (3, 3), 1, 0.1)
    conv2 = nn.Conv2d( intermediate_num_outputs,  num_outputs, (3, 3), padding=(1, 1), stride=1)

    return nn.Sequential(
        conv1,
        conv2
    )

def predict_motion_block( num_inputs , leaky_coef = 0.1):

    """
    :param num_inputs: number of input channels
    :return: rotation, translation and scale
    """

    """
    this block is --> (Conv+ReLU) --> FC --> FC --> FC -->,
    the output is rotation, translation and scale
    """

    conv1 = convrelu_block( num_inputs, 128,  (3, 3), 1, 0.1)

    fc1 = nn.Linear(128*8*6, 1024)
    fc2 = nn.Linear(1024, 128)
    fc3 = nn.Linear(128, 7)

    leaky_relu1 = nn.LeakyReLU(leaky_coef)
    leaky_relu2 = nn.LeakyReLU(leaky_coef)

    return conv1, \
           nn.Sequential(
               fc1,
               leaky_relu1,
               fc2,
               leaky_relu2,
               fc3)

class FlowBlock(nn.Module):

    def __init__(self, use_prev_predictions = False):

        super(FlowBlock, self).__init__()

        # self.conv1_1 = nn.Conv2d(6, 32, (9, 1), padding=(4, 0), stride=(2, 1) )
        # self.leaky_relu1_1 = nn.LeakyReLU(0.1)
        #
        # self.conv1_2 = nn.Conv2d(32, 32, (1, 9), padding=(0, 4), stride=(1, 2) )
        # self.leaky_relu1_2 = nn.LeakyReLU(0.1)
        #
        # self.conv1 = nn.Sequential(
        #     self.conv1_1,
        #     self.leaky_relu1_1,
        #     self.conv1_2,
        #     self.leaky_relu1_2)


        self.conv1 = convrelu2_block(6,  (32, 32), (9, 9), 2, 0.1)

        if(not use_prev_predictions):
            self.conv2 = convrelu2_block(32, (64, 64), (7, 7), 2, 0.1)
            self.conv2_1 = convrelu2_block(64, (64, 64), (3, 3), 1, 0.1)
        else:
            """ in this case we also use the information from previous depth prediction """
            self.warp_image = WarpImageLayer()
            self.depth_to_flow = DepthToFlowLayer(normalized_K=True)

            self.conv2 = convrelu2_block(32, (32, 32), (7, 7), 2, 0.1)
            self.conv2_extra_inputs = convrelu2_block(9, (32,32), (3, 3), 1, 0.1)
            self.conv2_1 = convrelu2_block(64, (64, 64), (3, 3), 1, 0.1)


        self.conv3 = convrelu2_block(64, (128,128), (5,5), 2, 0.1)
        self.conv3_1 = convrelu2_block(128, (128, 128), (3,3), 1, 0.1)

        self.conv4 = convrelu2_block(128, (256, 256), (5,5), 2, 0.1)
        self.conv4_1 = convrelu2_block(256, (256, 256), (3,3), 1, 0.1)


        """for conv5 layer, there is a mistake in the figure of demon paper, kernel size should be 5, not 3"""
        self.conv5 = convrelu2_block(256,(512, 512), (5,5), 2, 0.1)
        self.conv5_1 = convrelu2_block(512,(512, 512), (3,3), 1, 0.1)


        """five groups of convolution layers are finished"""

        self.flow1 = predict_flow_block(512, num_outputs=4)
        self.flow1_upconv = nn.ConvTranspose2d( 4, 2, (4,4), stride=(2,2), padding=1 )

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d( 512, 256, (4,4), stride=(2,2), padding=1),
            nn.LeakyReLU(0.1))

        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d( 514, 128, (4,4), stride=(2,2), padding=1),
            nn.LeakyReLU(0.1))

        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d( 256, 64,  (4,4), stride=(2,2), padding=1),
            nn.LeakyReLU(0.1))

        self.flow2 = predict_flow_block(128, num_outputs=4)


    def forward(self, image_pair, image2_2 = None, intrinsics = None, prev_predictions = None):
        """
        image_pair: Tensor
            Image pair concatenated along the channel axis.

        image2_2: Tensor
            Second image at resolution level 2 (downsampled two times)

        intrinsics: Tensor
            The normalized intrinsic parameters

        prev_predictions: dict of Tensor
            Predictions from the previous depth block
        """

        conv1 = self.conv1(image_pair)
        conv2 = self.conv2(conv1)

        if(prev_predictions == None):
            conv2_1 = self.conv2_1( conv2 )
        else:
            depth = prev_predictions['depth']
            normal = prev_predictions['normal']
            rotation = prev_predictions['rotation']
            translation = prev_predictions['translation']

            flow = self.depth_to_flow(intrinsics,
                                      intrinsics,
                                      depth,
                                      rotation,
                                      translation)

            warped_im = self.warp_image(image2_2, flow)
            combined = torch.cat((warped_im, flow, depth, normal), 1)

            """use torch.cat to concatenate tensors"""
            extra = self.conv2_extra_inputs( combined )
            conv2_1 = self.conv2_1( torch.cat((conv2, extra), 1) )

        conv3 = self.conv3( conv2_1 )
        conv3_1 = self.conv3_1( conv3 )
        conv4 = self.conv4( conv3_1 )
        conv4_1 = self.conv4_1( conv4 )
        conv5 = self.conv5( conv4_1 )
        conv5_1 = self.conv5_1( conv5 )

        upconv1 = self.upconv1(conv5_1)
        flow1 = self.flow1(conv5_1)
        flow1_upconv = self.flow1_upconv(flow1)

        """ concatenation along the channel axis """
        upconv2 = self.upconv2( torch.cat( (upconv1, conv4_1, flow1_upconv), 1 ) )
        upconv3 = self.upconv3( torch.cat( (upconv2, conv3_1), 1 ) )
        flow2 = self.flow2( torch.cat( (upconv3, conv2_1), 1) )

        """flow2 combines flow and flow confidence"""

        return flow2

""""
DepthMotionBlock is very similar to FlowBlock, probably we should just write a general one
"""
# class DepthMotionBlock(nn.Module):
class DepthMotionBlock(nn.Module):

    def __init__(self, use_prev_depthmotion = False):

        super(DepthMotionBlock, self).__init__()

        self.conv1 = convrelu2_block(6, (32,32), (9,9), 2, 0.1)
        self.conv2 = convrelu2_block(32,(32,32), (7,7), 2, 0.1)

        self.warp_image = WarpImageLayer()

        if(use_prev_depthmotion):
            self.conv2_extra_inputs = convrelu2_block(8, (32, 32), (3,3), 1, 0.1)
            self.flow_to_depth = FlowToDepthLayer(normalized_K=True)
        else:
            self.conv2_extra_inputs = convrelu2_block(7, (32, 32), (3,3), 1, 0.1)

        self.conv2_1 = convrelu2_block(64, (64, 64), (3,3), 1, 0.1)

        self.conv3 = convrelu2_block(64, (128,128), (5,5), 2, 0.1)
        self.conv3_1 = convrelu2_block(128, (128, 128), (3,3), 1, 0.1)

        self.conv4 = convrelu2_block(128, (256, 256), (5,5), 2, 0.1)
        self.conv4_1 = convrelu2_block(256, (256, 256), (3,3), 1, 0.1)

        """note that conv5 layer is different from FlowBlock, here the kernel size is 3"""
        self.conv5 = convrelu2_block(256,(512, 512), (3,3), 2, 0.1)
        self.conv5_1 = convrelu2_block(512,(512, 512), (3,3), 1, 0.1)

        self.motion_conv, self.motion_fc = predict_motion_block(512)

        """depth_normal predictione use the same architecture as predict_flow_block """
        self.depth_normal = predict_flow_block(128, num_outputs=4)

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d( 512, 256, (4,4), stride=(2,2), padding=1 ),
            nn.LeakyReLU(0.1))

        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d( 512, 128, (4,4), stride=(2,2), padding=1 ),
            nn.LeakyReLU(0.1))

        self.upconv3 = nn.Sequential(
            nn.ConvTranspose2d( 256, 64,  (4,4), stride=(2,2), padding=1 ),
            nn.LeakyReLU(0.1))

    def forward(self, image_pair, image2_2, prev_flow2, prev_flowconf2,
                prev_predictions = None, intrinsics = None):

        """
        image_pair: Tensor
            Image pair concatenated along the channel axis.

        image2_2: Tensor
            Second image at resolution level 2 (downsampled two times)

        prev_flow2: Tensor
            The output of the flow network. Contains only the flow (2 channels)

        prev_flowconf2: Tensor
            The output of the flow network. Contains flow and flow confidence (4 channels)

        prev_rotation: Tensor
            The previously predicted rotation.

        prev_translation: Tensor
            The previously predicted translation.

        intrinsics: Tensor
            The normalized intrinsic parameters

        """

        conv1 = self.conv1(image_pair)
        conv2 = self.conv2(conv1)

        """warp 2nd image"""
        warped_im = self.warp_image(image2_2, prev_flow2)

        if(prev_predictions == None):
            combined = torch.cat( (warped_im, prev_flowconf2), 1 )
        else:
            prev_rotation = prev_predictions['rotation']
            prev_translation = prev_predictions['translation']
            depth = self.flow_to_depth(intrinsics, intrinsics, prev_flow2, prev_rotation, prev_translation)

            combined = torch.cat((warped_im, prev_flowconf2, depth), 1)

            # """  testing """
            # combined = torch.cat((warped_im, prev_flowconf2, warped_im[0:1,0:1,:,:]), 1)

        extra = self.conv2_extra_inputs( combined )
        conv2_1 = self.conv2_1( torch.cat((conv2, extra),1) )

        conv3 = self.conv3( conv2_1 )
        conv3_1 = self.conv3_1( conv3 )
        conv4 = self.conv4( conv3_1 )
        conv4_1 = self.conv4_1( conv4 )
        conv5 = self.conv5( conv4_1 )
        conv5_1 = self.conv5_1( conv5 )

        upconv1 = self.upconv1(conv5_1)
        upconv2 = self.upconv2( torch.cat((upconv1, conv4_1), 1) )
        upconv3 = self.upconv3( torch.cat((upconv2, conv3_1), 1) )

        depth_normal = self.depth_normal( torch.cat((upconv3, conv2_1), 1) )
        motion_conv = self.motion_conv(conv5_1)

        motion = self.motion_fc(
            motion_conv.view(
                motion_conv.size(0),
                128 * 6 * 8
        ))

        scale = motion[:,6]
        rotation = motion[:,0:3]
        translation = motion[:,3:6]

        # batch_size = scale.size(0)

        depth = depth_normal[:, 0:1, :, :] * scale.expand_as( depth_normal[:, 0:1, :, :] )
        normal = depth_normal[:, 1:4, :, :]

        predictions = {
            'depth': depth,
            'normal': normal,
            'rotation': rotation,
            'translation': translation,
        }

        return predictions

"""
Refinement Block
"""
class RefinementBlock(nn.Module):

    def __init__(self):

        super(RefinementBlock, self).__init__()

        self.conv0 = convrelu_block(4, 32, (3,3), (1,1), 0.1)
        self.conv1 = convrelu_block(32, 64, (3,3), (2,2), 0.1)
        self.conv1_1 = convrelu_block(64, 64, (3,3), (1,1), 0.1)

        self.conv2 = convrelu_block(64, 128, (3,3), (2,2), 0.1)
        self.conv2_1 = convrelu_block(128, 128, (3,3), (1,1), 0.1)

        self.upconv1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, (4,4), stride=(2,2), padding=1),
            nn.LeakyReLU(0.1)
        )

        self.upconv2 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, (4,4), stride=(2,2), padding=1),
            nn.LeakyReLU(0.1)
        )

        self.depth_refine = predict_flow_block(64, num_outputs=1, intermediate_num_outputs=16)

    def forward(self, image1, depth):

        """
        :param image1:
        :param depth:
        :return:
        """

        """
        fix me, update upsampling
        """
        W = image1.shape[-1]
        H = image1.shape[-2]

        up_sample = nn.UpsamplingNearest2d(size=(H, W))
        depth_upsampled = up_sample(depth)

        input = torch.cat(
            (
                torch.autograd.Variable(
                    torch.from_numpy(image1),
                    requires_grad = False),
                depth_upsampled),
            1)

        conv0 = self.conv0(input)
        conv1 = self.conv1(conv0)
        conv1_1 = self.conv1_1(conv1)

        conv2 = self.conv2(conv1_1)
        conv2_1 = self.conv2_1(conv2)

        upconv1 = self.upconv1(conv2_1)
        upconv2 = self.upconv2( torch.cat((upconv1, conv1_1), 1) )

        depth_refine = self.depth_refine( torch.cat((upconv2, conv0), 1) )

        return depth_refine


