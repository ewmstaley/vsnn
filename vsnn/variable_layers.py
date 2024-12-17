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
import math
import copy
from vsnn.masks import MaskMakerSingleton
from vsnn.utils import extractSublayerFromLinear
from vsnn.utils import extractSublayerFromConv2d
from vsnn.utils import extractSublayerFromConv3d


# Superclass for a generic module which could be a layer or something more complex.
class VariableModule(torch.nn.Module):

    def __init__(self, max_size):
        super().__init__()
        self.max_size = max_size
        self.masker = MaskMakerSingleton()
        def reseed_masker_hook(module, grad_input, grad_output):
            module.masker.reseed()
        if not self.masker.external_reseed:
            self.register_backward_hook(reseed_masker_hook)


# Superclass for a generic layer that uses triangular dropout along some axis.
class VariableLayer(VariableModule):

    def __init__(self, layer, max_size, axis):
        super().__init__(max_size)
        self.submodules = torch.nn.ModuleList([layer])
        self.axis = axis

    def forward(self, x, mask_mode=-1, diagonal=0):
        y = self.submodules[0](x)
        with torch.no_grad():
            mask = self.masker.make_mask(y, self.max_size, mask_mode=mask_mode, axis=self.axis, diagonal=diagonal)
        y = y*mask
        return y

    def staticSublayer(self, test_mask_mode):
        # get part of this layer as a new standalone layer
        raise NotImplementedError

    def mask_mode_as_integer(self, mask_mode):
        # for convenience, convert mask mode to an interger
        raise NotImplementedError


class VariableLinear(VariableLayer):
    def __init__(self, in_features, out_features, **kwargs):
        self.in_features = in_features
        self.out_features = out_features
        layer = torch.nn.Linear(in_features, out_features)
        super().__init__(layer, out_features, axis=-1)

    def staticSublayer(self, test_mask_mode, in_features):
        return extractSublayerFromLinear(self.submodules[0], test_mask_mode, in_features)

    def mask_mode_as_integer(self, mask_mode):
        if isinstance(mask_mode, float):
            mask_mode = int(round(mask_mode*float(self.out_features)))
        return mask_mode 


class VariableConv2d(VariableLayer):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, kernel_size
        self.kwargs = kwargs
        layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        super().__init__(layer, out_channels, axis=1)

    def staticSublayer(self, test_mask_mode, in_ch):
        return extractSublayerFromConv2d(self.submodules[0], test_mask_mode, in_ch, **self.kwargs)

    def mask_mode_as_integer(self, mask_mode):
        if isinstance(mask_mode, float):
            mask_mode = int(round(mask_mode*float(self.out_channels)))
        return mask_mode 


class VariableConv3d(VariableLayer):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        self.in_channels, self.out_channels, self.kernel_size = in_channels, out_channels, kernel_size
        self.kwargs = kwargs
        layer = torch.nn.Conv3d(in_channels, out_channels, kernel_size, **kwargs)
        super().__init__(layer, out_channels, axis=1)

    def staticSublayer(self, test_mask_mode, in_ch):
        return extractSublayerFromConv3d(self.submodules[0], test_mask_mode, in_ch, **self.kwargs)

    def mask_mode_as_integer(self, mask_mode):
        if isinstance(mask_mode, float):
            mask_mode = int(round(mask_mode*float(self.out_channels)))
        return mask_mode 