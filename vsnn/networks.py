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

'''
Some utilities to make simple MLPs.
'''

import torch
from vsnn.variable_layers import VariableLinear

class VMLP(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, hidden_layers=2):
        super().__init__()
        assert hidden_layers>0, "must have at least one hidden layer"
        self.input_size, self.output_size = input_size, output_size
        self.vlayers = torch.nn.ModuleList()
        self.vlayers.append(VariableLinear(input_size, hidden_size))
        for i in range(hidden_layers):
            self.vlayers.append(VariableLinear(hidden_size, hidden_size))
        self.vlayers.append(VariableLinear(hidden_size, output_size))

    def forward(self, x, mask=-1, diagonal=0):
        for layer in self.vlayers[:-1]:
            x = torch.nn.functional.relu(layer(x, mask, diagonal))
        x = self.vlayers[-1](x, 1.0)
        return x

    def getStaticSubnet(self, test_mask_mode):
        subnet = torch.nn.Sequential()
        prev_size = self.input_size
        for layer in self.vlayers[:-1]:
            new_layer = layer.staticSublayer(test_mask_mode, prev_size)
            prev_size = layer.mask_mode_as_integer(test_mask_mode)
            subnet.append(new_layer)
            subnet.append(torch.nn.ReLU())
        subnet.append(self.vlayers[-1].staticSublayer(1.0, prev_size))
        subnet.to(next(self.parameters()).device) # put subnet on same device as this module
        return subnet


# for comparisons to VMLP
class StandardMLP(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size, hidden_layers=2):
        super().__init__()
        assert hidden_layers>0, "must have at least one hidden layer"
        self.input_size, self.output_size = input_size, output_size
        self.layers = torch.nn.ModuleList()
        self.layers.append(torch.nn.Linear(input_size, hidden_size))
        for i in range(hidden_layers):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
        self.layers.append(torch.nn.Linear(hidden_size, output_size))

    def forward(self, x, mask=None):
        for layer in self.layers[:-1]:
            x = torch.nn.functional.relu(layer(x))
        x = self.layers[-1](x)
        return x
