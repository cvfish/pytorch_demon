import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import os
import sys

from torch.autograd import Variable

from python.demon_networks import *
from util import load_parameters

# insert this to the top of your scripts (usually main.py)
import sys, warnings, traceback, torch
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
    traceback.print_stack(sys._getframe(2))
warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
torch.utils.backcompat.broadcast_warning.enabled = False
torch.utils.backcompat.keepdim_warning.enabled = True


examples_dir = os.path.dirname(__file__)
weights_dir = os.path.join(examples_dir,'..','weights')
sys.path.insert(0, os.path.join(examples_dir, '..', 'python'))

def prepare_input_data(img1, img2, data_format):
    """Creates the arrays used as input from the two images."""
    # scale images if necessary
    if img1.size[0] != 256 or img1.size[1] != 192:
        img1 = img1.resize((256,192))
    if img2.size[0] != 256 or img2.size[1] != 192:
        img2 = img2.resize((256,192))
    img2_2 = img2.resize((64,48))
        
    # transform range from [0,255] to [-0.5,0.5]
    img1_arr = np.array(img1).astype(np.float32)/255 -0.5
    img2_arr = np.array(img2).astype(np.float32)/255 -0.5
    img2_2_arr = np.array(img2_2).astype(np.float32)/255 -0.5
    
    if data_format == 'channels_first':
        img1_arr = img1_arr.transpose([2,0,1])
        img2_arr = img2_arr.transpose([2,0,1])
        img2_2_arr = img2_2_arr.transpose([2,0,1])
        image_pair = np.concatenate((img1_arr,img2_arr), axis=0)
    else:
        image_pair = np.concatenate((img1_arr,img2_arr),axis=-1)
    
    result = {
        'image_pair': image_pair[np.newaxis,:],
        'image1': img1_arr[np.newaxis,:], # first image
        'image2_2': img2_2_arr[np.newaxis,:], # second image with (w=64,h=48)
    }
    return result

# 
# DeMoN has been trained for specific internal camera parameters.
#
# If you use your own images try to adapt the intrinsics by cropping
# to match the following normalized intrinsics:
#
#  K = (0.89115971  0           0.5)
#      (0           1.18821287  0.5)
#      (0           0           1  ),
#  where K(1,1), K(2,2) are the focal lengths for x and y direction.
#  and (K(1,3), K(2,3)) is the principal point.
#  The parameters are normalized such that the image height and width is 1.
#

K = [[0.89115971,  0,  0.5],
     [0,  1.18821287,  0.5],
     [0,           0,    1]]
intrinsics = Variable(torch.Tensor( K ), requires_grad=False)

# read data
img1 = Image.open(os.path.join(examples_dir,'sculpture1.png'))
img2 = Image.open(os.path.join(examples_dir,'sculpture2.png'))

input_data = prepare_input_data(img1,img2,'channels_first')

""" the whole network """
bootstrap_net = BootstrapNet()
iterative_net = IterativeNet()
refinement_net = RefinementNet()

"""load parameters"""
bootstrap_net.load_state_dict(torch.load('./bootstrap_net.pt'))
iterative_net.load_state_dict(torch.load('./iterative_net.pt'))
refinement_net.load_state_dict(torch.load('./refinement_net.pt'))

# run the bootstrap net and 3 times iterative net
img_pair = Variable( torch.FloatTensor(input_data['image_pair']), requires_grad=False )
img2 = Variable( torch.FloatTensor(input_data['image2_2']), requires_grad=False )

result = bootstrap_net(img_pair, img2)

for i in range(3):
    result = iterative_net(
        img_pair,
        img2,
        intrinsics,
        result
    )

# run refinemnt net to refine and increase depth map resolution
result = refinement_net(input_data['image1'], result['depth'])