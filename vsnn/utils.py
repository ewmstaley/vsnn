'''
Copyright © 2024 The Johns Hopkins University Applied Physics Laboratory LLC
 
Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.
 
THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import torch
import copy

# ================================================================================
def count_params(model):
    return sum(p.numel() for p in model.parameters())

# ================================================================================
# These utilities create a new layer object which is smaller than some
# reference layer and uses a subportion of the original layer's weights.
# For example, starting from a Linear(N,M) layer, we could get a sublayer
# with any dimensions (n,m) in (1<=n<=N, 1<=m<=M)

# get a sublayer from a linear layer
def extractSublayerFromLinear(layer, test_mask_mode, in_features):
    out_features = layer.weight.shape[0]
    if isinstance(test_mask_mode, float):
       test_mask_mode = int(round(test_mask_mode*float(out_features)))
    m = max(min(out_features, test_mask_mode), 1)
    new_layer = torch.nn.Linear(in_features, m)
    new_layer.weight.data[:,:] = copy.deepcopy(layer.weight.data[:m,:in_features])
    new_layer.bias.data[:] = copy.deepcopy(layer.bias.data[:m])
    return new_layer

# get a sublayer from a conv2d layer
def extractSublayerFromConv2d(layer, test_mask_mode, in_ch, **kwargs):
    out_ch = layer.weight.shape[0]
    kernel = layer.weight.shape[2] # assuming square kernels
    if isinstance(test_mask_mode, float):
       test_mask_mode = int(round(test_mask_mode*float(out_ch)))
    m = max(min(out_ch, test_mask_mode), 1)
    new_layer = torch.nn.Conv2d(in_ch, m, kernel, **kwargs)
    new_layer.weight.data[:,:,:,:] = copy.deepcopy(layer.weight.data[:m,:in_ch,:,:])
    new_layer.bias.data[:] = copy.deepcopy(layer.bias.data[:m])
    return new_layer

# get a sublayer from a conv3d layer
def extractSublayerFromConv3d(layer, test_mask_mode, in_ch, **kwargs):
    out_ch = layer.weight.shape[0]
    kernel = layer.weight.shape[2] # assuming cube kernels
    if isinstance(test_mask_mode, float):
       test_mask_mode = int(round(test_mask_mode*float(out_ch)))
    m = max(min(out_ch, test_mask_mode), 1)
    new_layer = torch.nn.Conv3d(in_ch, m, kernel, **kwargs)
    new_layer.weight.data[:,:,:,:,:] = copy.deepcopy(layer.weight.data[:m,:in_ch,:,:,:])
    new_layer.bias.data[:] = copy.deepcopy(layer.bias.data[:m])
    return new_layer

