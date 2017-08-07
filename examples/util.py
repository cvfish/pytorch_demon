"""
loading provided parameters from demon
"""

import pickle

import torch
import numpy as np


def load_parameters(filename, network):

    with open(filename, 'rb') as f:
        parameters = pickle.load(f)

    bootstrap_net, iterative_net, refinement_net = network

    flow_dict, extra_flow_dict, depth_motion_dict, refine_dict = parameter_mapping_dict()

    """copy parameters to bootstrap_net"""
    num1 = copy_weights(parameters, bootstrap_net, flow_dict, prefix = 'netFlow1/')
    num2 = copy_weights(parameters, bootstrap_net, depth_motion_dict, prefix = 'netDM1/')

    """copy parameters to iterative_net"""
    num3 = copy_weights(parameters, iterative_net, flow_dict, prefix='netFlow2/')
    num4 = copy_weights(parameters, iterative_net, extra_flow_dict, prefix='netFlow2/')
    num5 = copy_weights(parameters, iterative_net, depth_motion_dict, prefix='netDM2/')

    """copy parameters to refinement_net"""
    num6 = copy_weights(parameters, refinement_net, refine_dict, prefix = 'netRefine/')

    print "total number of weight: {:03d}".format(num1 + num2 + num3 + num4 + num5 + num6)

def copy_weights(parameters_tf, net, dict, prefix):

    num_weights_copied = 0

    temp_dict = net.state_dict()
    
    for key, value in dict.iteritems():

        """
        check first if parameters belong to convolution layer or not
        """

        tf_shape = parameters_tf[prefix + key].shape
        # py_shape = net.state_dict()[value].size()
        py_shape = temp_dict[value].size()

        print prefix + key
        print tf_shape
        print value
        print py_shape

        if 'fc' not in key:

            if(len(tf_shape) == 1):  ### bias
                assert tf_shape[0] == py_shape[0]
                temp = parameters_tf[prefix + key].copy()
                num_weights_copied += 1
            else:  ### weight
                assert tf_shape[3] == py_shape[0] and tf_shape[2] == py_shape[1]
                assert tf_shape[0] == py_shape[2] and tf_shape[1] == py_shape[3]
                temp = np.transpose(parameters_tf[prefix + key], (3, 2, 0, 1)).copy()
                num_weights_copied += 1

        else:

            if (len(tf_shape) == 1):   ### scalar output
                assert tf_shape[0] == py_shape[0]
                temp = parameters_tf[prefix + key].copy()
                num_weights_copied += 1
            else: ### vector output
                assert tf_shape[1] == py_shape[0] and tf_shape[0] == py_shape[1]
                temp = np.transpose(parameters_tf[prefix + key], (1, 0)).copy()
                num_weights_copied += 1

        np.ascontiguousarray(temp, dtype=np.float32)
        temp_dict[value] = torch.from_numpy(temp)

    ### load state_dict
    net.load_state_dict(temp_dict)

    return num_weights_copied

def parameter_mapping_dict():

    """
    map parameters from tensorflow to pytorch
    """

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    flow block
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    flow_dict = {}

    for conv in ['conv1', 'conv2', 'conv2_1', 'conv3', 'conv3_1', 'conv4', 'conv4_1', 'conv5', 'conv5_1']:

        flow_dict[conv + 'y/bias:0'] = 'flow_block.' + conv + '.0.bias'
        flow_dict[conv + 'y/kernel:0'] = 'flow_block.' + conv + '.0.weight'
        flow_dict[conv + 'x/bias:0'] = 'flow_block.' + conv + '.2.bias'
        flow_dict[conv + 'x/kernel:0'] = 'flow_block.' + conv + '.2.weight'

    for flow in ['flow1', 'flow2']:

        if flow == 'flow1':
            prefix = 'predict_flow5'
        elif flow == 'flow2':
            prefix = 'predict_flow2'

        flow_dict[prefix + '/conv1/' + 'bias:0'] = 'flow_block.' + flow + '.0.0.bias'
        flow_dict[prefix + '/conv1/' + 'kernel:0'] = 'flow_block.' + flow + '.0.0.weight'

        flow_dict[prefix + '/conv2/' + 'bias:0'] = 'flow_block.' + flow + '.1.bias'
        flow_dict[prefix + '/conv2/' + 'kernel:0'] = 'flow_block.' + flow + '.1.weight'

    upconv = {"refine4":"upconv1", "refine3":"upconv2", "refine2":"upconv3"}

    for key, value in upconv.iteritems():

        flow_dict[key + '/upconv/' + 'bias:0'] = 'flow_block.' + value + '.0.bias'
        flow_dict[key + '/upconv/' + 'kernel:0'] = 'flow_block.' + value + '.0.weight'

    flow_dict['upsample_flow5to4/upconv/bias:0'] = 'flow_block.flow1_upconv.bias'
    flow_dict['upsample_flow5to4/upconv/kernel:0'] = 'flow_block.flow1_upconv.weight'

    extra_flow_dict = {}

    for conv in ['conv2_extra_inputs']:

        extra_flow_dict[conv + 'y/bias:0'] = 'flow_block.' + conv + '.0.bias'
        extra_flow_dict[conv + 'y/kernel:0'] = 'flow_block.' + conv + '.0.weight'

        extra_flow_dict[conv + 'x/bias:0'] = 'flow_block.' + conv + '.2.bias'
        extra_flow_dict[conv + 'x/kernel:0'] = 'flow_block.' + conv + '.2.weight'

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    depth motion block
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    depth_motion_dict = {}

    for conv in ['conv1', 'conv2', 'conv2_1', 'conv3', 'conv3_1',
                 'conv4', 'conv4_1', 'conv5', 'conv5_1', 'conv2_extra_inputs']:

        depth_motion_dict[conv + 'y/bias:0'] =   'depth_motion_block.' + conv + '.0.bias'
        depth_motion_dict[conv + 'y/kernel:0'] = 'depth_motion_block.' + conv + '.0.weight'
        depth_motion_dict[conv + 'x/bias:0'] =   'depth_motion_block.' + conv + '.2.bias'
        depth_motion_dict[conv + 'x/kernel:0'] = 'depth_motion_block.' + conv + '.2.weight'

    motion = {"motion_conv1": "_conv.0",
              "motion_fc1": "_fc.0",
              "motion_fc2": "_fc.2",
              "motion_fc3": "_fc.4" }

    for key, value in motion.iteritems():

        depth_motion_dict[key + '/bias:0'] = 'depth_motion_block.motion' + value + '.bias'
        depth_motion_dict[key + '/kernel:0'] = 'depth_motion_block.motion' + value + '.weight'

    depth_normal = {"conv1": "0.0", "conv2":"1"}

    for key, value in depth_normal.iteritems():

        depth_motion_dict['predict_depthnormal2/' + key + '/bias:0'] = 'depth_motion_block.depth_normal.' + value + '.bias'
        depth_motion_dict['predict_depthnormal2/' + key + '/kernel:0'] = 'depth_motion_block.depth_normal.' + value + '.weight'

    upconv = {"refine4":"upconv1", "refine3":"upconv2", "refine2":"upconv3"}

    for key, value in upconv.iteritems():

        depth_motion_dict[key + '/upconv/' + 'bias:0'] = 'depth_motion_block.' + value + '.0.bias'
        depth_motion_dict[key + '/upconv/' + 'kernel:0'] = 'depth_motion_block.' + value + '.0.weight'

    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    refine block
    """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    refine_dict = {}

    for conv in ['conv0', 'conv1', 'conv1_1', 'conv2', 'conv2_1']:

        refine_dict[conv + '/bias:0'] =   'refinement_block.' + conv + '.0.bias'
        refine_dict[conv + '/kernel:0'] = 'refinement_block.' + conv + '.0.weight'

    upconv = {"refine1": "upconv1", "refine0": "upconv2"}

    for key, value in upconv.iteritems():

        refine_dict[key + '/upconv/' + 'bias:0'] = 'refinement_block.' + value + '.0.bias'
        refine_dict[key + '/upconv/' + 'kernel:0'] = 'refinement_block.' + value + '.0.weight'

    depth_refine = {"conv1": "0.0", "conv2":"1"}

    for key, value in depth_refine.iteritems():

        refine_dict['predict_depth0/' + key + '/bias:0'] = 'refinement_block.depth_refine.' + value + '.bias'
        refine_dict['predict_depth0/' + key + '/kernel:0'] = 'refinement_block.depth_refine.' + value + '.weight'

    return flow_dict, extra_flow_dict, depth_motion_dict, refine_dict

