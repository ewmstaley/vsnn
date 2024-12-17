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
Simple example of training an MLP with variable-width layers.
An MLP is trained to match the outputs of another (frozen, randomly initialized) MLP.
'''

import torch
from vsnn.variable_layers import VariableLinear
from vsnn.masks import MaskMakerSingleton
import matplotlib.pyplot as plt
import numpy as np
import time

input_size = 32
output_size = 32
TARG_W = 64
MODEL_W = 256

np.random.seed(1337)
torch.random.manual_seed(1337)

# target model
class Target(torch.nn.Module):

	def __init__(self):
		super().__init__()
		w = TARG_W
		self.fc1 = torch.nn.Linear(input_size, w)
		self.fc2 = torch.nn.Linear(w, w)
		self.fc3 = torch.nn.Linear(w, w)
		self.fc4 = torch.nn.Linear(w, w)
		self.fc5 = torch.nn.Linear(w, output_size)

		for layer in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
			torch.nn.init.uniform_(layer.weight, a=-1.0, b=1.0)
			torch.nn.init.uniform_(layer.bias, a=-1.0, b=1.0)

	def forward(self, x):
		x = torch.nn.functional.tanh(self.fc1(x))
		x = torch.nn.functional.tanh(self.fc2(x))
		x = torch.nn.functional.tanh(self.fc3(x))
		x = torch.nn.functional.tanh(self.fc4(x))
		x = self.fc5(x)
		return x


# our model
class VMLP(torch.nn.Module):

	def __init__(self):
		super().__init__()
		w = MODEL_W
		self.vc1 = VariableLinear(input_size, w)
		self.vc2 = VariableLinear(w, w)
		self.vc3 = VariableLinear(w, w)
		self.output_layer = torch.nn.Linear(w, output_size)

	def forward(self, x, mask=-2):
		x = torch.nn.functional.relu(self.vc1(x, mask))
		x = torch.nn.functional.relu(self.vc2(x, mask))
		x = torch.nn.functional.relu(self.vc3(x, mask))
		x = self.output_layer(x)
		return x


# train
device = torch.device("cuda")
target_function = Target().to(device)

# edit some of the masker properties
masker = MaskMakerSingleton()
masker.external_reseed = True

vmlp = VMLP().to(device)
vmlp = torch.compile(vmlp)

opt = torch.optim.Adam(vmlp.parameters(), lr=0.0005)

st = time.time()
losses = []
for i in range(1000):
	st2 = time.time()
	opt.zero_grad()
	masker.reseed()
	inp = torch.rand((MODEL_W*20, input_size)).to(device)
	with torch.no_grad():
		targ = target_function(inp)

	# we pass "training_random" to randomly mask parts of the network,
	# simulating many different (sub)network sizes.
	# See vsnn/masks.py for more details.
	outp = vmlp(inp, mask="training_random")

	loss = torch.nn.functional.mse_loss(outp, targ)
	print(i, loss, time.time()-st2)
	losses.append(loss.item())
	loss.backward()
	opt.step()

print("Elapsed time:", time.time()-st)

# ==========================================
# Evaluation
vmlp.eval()
xs = np.linspace(0.0, 1.0, num=500).tolist()
with torch.no_grad():
	inp = torch.rand((MODEL_W*10, input_size)).to(device)
	targ = target_function(inp)

	ys = []
	for frac in xs:

		# for testing, we can pass a fractional value for how much of the layer
		# we want to retain (in terms of width). This effectively ablates neurons
		# from the output of the layer.
		# See vsnn/masks.py for more details.
		masker.reseed()
		outp = vmlp(inp, mask=frac)
		loss = torch.nn.functional.mse_loss(outp, targ)
		ys.append(loss.cpu().data.numpy())

# show performance as a function of model width
plt.plot(xs, ys)
plt.xlabel("Hidden Layer Width (Fraction)")
plt.ylabel("Evaluation Loss")
plt.show()