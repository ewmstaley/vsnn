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
import numpy as np

'''
Make a mask for a given batch, with masking varying along a particular axis.
The first axis is always assumed to be the batch axis.
There are a variety of behaviors depending on the batch size and mask mode:

mask_mode = -1 or "train": (default) Triangular Dropout training case. There are three sub-cases:
    1. The batch size == max_size. A perfect tringular matrix is formed. 
    For example, batch size == max_size == 4:
    1 0 0 0
    1 1 0 0
    1 1 1 0
    1 1 1 1

    2. The batch size is less than max_size and evenly divides it. A block
    triangular matrix is used. For example, batch=3, max_size=6:
    1 1 0 0 0 0
    1 1 1 1 0 0
    1 1 1 1 1 1

    3. The batch size is greater than max_size and is an exact multiple of it.
    A triangular block matrix is used. For example, batch=6, max_size=3:
    1 0 0
    1 0 0
    1 1 0
    1 1 0
    1 1 1
    1 1 1

    If the batch size does not evenly divide the max_size, or vice versa, an error is thrown.

mask_mode = -2 or "train_random":
    Alternative training case. Random masks are created that fit the pattern of:
    [random amount of 1s][rest 0s]

    This does not have a batch size constraint and therefore can be used for arbitrary training cases.

    Must call reseed() on the mask maker singleton between batches (done automatically by layers)

-3 < mask_mode < -2:
    Converted to a fraction: (-m)-2, i.e. [-2.1, -2.3, -2.5] map to [0.1, 0.3, 0.5]
    This designates a fraction of samples that should be unmasked, so that we can
    interpolate (in expectation) between triangular dropout and no dropout.
    The behavior is otherwise the same as mask_mode=-2.

    For example, mask_mode=-2.25 will randomly sample triangular dropout masks 75% of
    the time, and use no mask 25% of the time.

mask_mode >= 1, integer: 
    Retain a specific number of output nodes.
    For example, a linear layer of width 4 with mask_mode=3. This will
    retain the first 3 nodes, with sample-wise masks of: [1,1,1,0].
    Note this is clamped to [1, max_size], inclusive.

0.0 >= mask_mode >= 1.0, float: 
    Retain a specific fraction of output nodes.
    This is first rounded to the nearest whole node.
    For example, a linear layer of width 4 with mask_mode=0.7. First, we have
    0.7*4 = 2.8. This is rounded to 3, and we then get sample-wise masks
    of [1,1,1,0]. Note this is clamped to [1, max_size], inclusive.

mask_mode = None
    Mask is all ones (equivalent to no masking, or mask_mode=1.0)

mask_mode = method
    A method is supplied which returns value for all inputs [0.0 - 1.0] inclusive.
    This is used to weight the probability of masks with fractional size 0.0 to 1.0.


=====================================================================

IMPORTANT NOTE:
To support multiple sizes, set masker.base to the greatest common divisor of all sizes.
Masks will then be built for this size and tiled to the appropriate final size.
This ensures that all masks are compatible with one another.

For example, imagine we have a network with layers of width 8 and 12.
We set masker.base = 4
Now, masks are constructed for a width of four:
[1, 1, 1, 0]

And repeated to the appropriate size. Width-8 would see:
[1, 1, 1, 1, 1, 1, 0, 0]

and width-12 would see:
[1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]

'''

class MaskMakerSingleton():

    def __new__(cls):
        if not hasattr(cls, 'instance'):
          cls.instance = super(MaskMakerSingleton, cls).__new__(cls)
          cls.instance.seed = 1337
          cls.instance.base = None
          cls.instance.cache = {}
          cls.external_reseed = False
        return cls.instance

    def reseed(self, seed=None):
        if seed is not None:
            self.seed = seed
        else:
            self.seed = np.random.randint(9999999, 99999999999)
        self.cache = {}
        if self.external_reseed:
            # reseed here instead of within the mask generation
            # requires us to use the cache mechanism
            torch.random.fork_rng()
            torch.random.manual_seed(self.seed)

    def make_mask(
        self,
        batch,
        max_size,
        mask_mode=-1,
        axis=1,
        diagonal=0
    ):

        assert len(batch.size())>=2, "Input must have a separate batch dimension."
        assert axis != 0, "Batch dimension is given axis 0. Please use a different axis."

        if axis < 0:
            axis = len(batch.size())+axis

        B = batch.size()[0]
        N = max_size

        if self.base is not None:
            N = self.base
            assert max_size%self.base == 0, "If using masker.base, all sizes must be divisible by base."

        if isinstance(mask_mode, str):
            assert mask_mode in ["training", "train", "training_random", "train_random"], "mask must be a number or one of: training, train, training_random, train_random"
            if mask_mode in ["training", "train"]:
                mask_mode = -1
            elif mask_mode in ["training_random", "train_random"]:
                mask_mode = -2


        if (B,N,mask_mode,type(mask_mode)) in self.cache:
            mask = self.cache[(B,N,mask_mode,type(mask_mode))]
        else:

            # identity mask
            if mask_mode is None:
                mask = torch.ones((B,N), device=batch.device)

            # fully custom mapping from [0-1] inclusive to relative probability weights
            elif callable(mask_mode):
                if not self.external_reseed:
                    torch.random.fork_rng()
                    torch.random.manual_seed(self.seed)

                # make perfect mask
                mask = torch.ones((N,N), device=batch.device)
                mask = torch.tril(mask, diagonal=diagonal)

                # get each row as a fraction [0-1]
                fracs = torch.linspace(0.0, 1.0, N)

                # get some relative weight for each fraction
                # proportional to the chance of sampling that mask
                weights = mask_mode(fracs)

                # sample B masks
                idxs = torch.multinomial(weights, B, replacement=True)

                # get output
                mask = mask[idxs]

            # training case (default)
            elif mask_mode == -1:

                if B==N:
                    # case 1, perfect batch size
                    mask = torch.ones((N,N), device=batch.device)
                    mask = torch.tril(mask, diagonal=diagonal)

                elif B<N:
                    # block matrix, batch size too small
                    assert N%B==0, "Batch size must be an exact multiple or divisor of max size"
                    mask = torch.ones((B,B), device=batch.device)
                    mask = torch.tril(mask, diagonal=diagonal)
                    mask = torch.repeat_interleave(mask, N//B, dim=1)

                else:
                    # block matrix, batch size too large
                    assert B%N==0, "Batch size must be an exact multiple or divisor of max size"
                    mask = torch.ones((N,N), device=batch.device)
                    mask = torch.tril(mask, diagonal=diagonal)
                    mask = torch.repeat_interleave(mask, B//N, dim=0)

            # training case: randomized
            elif mask_mode == -2:
                if not self.external_reseed:
                    torch.random.fork_rng()
                    torch.random.manual_seed(self.seed)

                # make perfect mask
                mask = torch.ones((N,N), device=batch.device)
                mask = torch.tril(mask, diagonal=diagonal)

                if B<=N:
                    # choose B rows
                    indexes = torch.randperm(N)[:B]
                    mask  = mask[indexes]
                else:
                    # repeat indicies and then choose B rows 
                    repeats = B//N + 1
                    num_choices = N*repeats
                    choices = torch.arange(num_choices)%N
                    index_perm = torch.randperm(num_choices)
                    indexes = choices[index_perm]
                    indexes = indexes[:B]
                    mask  = mask[indexes]


            elif mask_mode < -2:
                frac_no_mask = (-mask_mode) - 2.0

                if not self.external_reseed:
                    torch.random.fork_rng()
                    torch.random.manual_seed(self.seed)

                # make perfect mask
                mask = torch.ones((N,N), device=batch.device)
                mask = torch.tril(mask, diagonal=diagonal)

                # same as -2 case above
                if B<=N:
                    # choose B rows
                    indexes = torch.randperm(N)[:B]
                    mask  = mask[indexes]
                else:
                    # repeat indicies and then choose B rows 
                    repeats = B//N + 1
                    num_choices = N*repeats
                    choices = torch.arange(num_choices)%N
                    index_perm = torch.randperm(num_choices)
                    indexes = choices[index_perm]
                    indexes = indexes[:B]
                    mask  = mask[indexes]

                # for each mask entry, there is a [frac] chance that it will be replaced with ones
                rand_unmask = torch.rand(mask.shape[0]).to(batch.device)
                rand_unmask = torch.where(rand_unmask < frac_no_mask, 1, 0)
                rand_unmask = rand_unmask[:,None].repeat(1, N)
                unmask = torch.ones_like(mask).to(batch.device)

                mask = mask*(1.0-rand_unmask) + unmask*rand_unmask


            # specific-width case
            else:
                # convert to int if needed
                if isinstance(mask_mode, float):
                   mask_mode = int(round(mask_mode*float(N)))

                # clamp
                m = max(min(N, mask_mode), 1)

                mask = torch.ones((B,N), device=batch.device)
                mask[:,m:] = 0.0

            # cache mask
            self.cache[(B,N,mask_mode,type(mask_mode))] = torch.clone(mask)

        # we now have a mask of size (B,N)

        # if N==self.base, we may need to expand to max_size
        if N != max_size:
            mask = torch.repeat_interleave(mask, max_size//self.base, dim=-1)

        # we now have a mask of size (B,max_size) that may need more dimensions
        # how many dimensions do we need?
        dims = len(batch.size())

        # add extra trailing dims
        for i in range(dims-2):
            mask = mask.unsqueeze(-1)

        # transpose so mask dimension is on correct axis
        order = list(range(dims))
        if axis != 1:
            order[1], order[axis] = order[axis], order[1]
            mask = torch.permute(mask, order)

        # repeat dimensions until same size as batch
        repeats = list(batch.size())
        repeats[0] = 1
        repeats[axis] = 1
        mask = mask.repeat(repeats)

        # all done!
        return mask




if __name__ == "__main__":

    masker = MaskMakerSingleton()
    masker.reseed()
    batch = torch.ones((20,8))

    masker.base = 4
    mask = masker.make_mask(batch, 8, mask_mode=-2.9)
    print(mask)

    # print("\nShould be different:............")
    # print("-")
    # mask = masker.make_mask(batch, 8, -2)
    # print(mask)

    # batch = torch.ones((5,12))
    # mask = masker.make_mask(batch, 12, -2)
    # print(mask)

    # print("\nShould be similar:............")
    # masker.base = 4
    # batch = torch.ones((5,8))
    # print("-")
    # mask = masker.make_mask(batch, 8, -2)
    # print(mask)

    # batch = torch.ones((5,12))
    # mask = masker.make_mask(batch, 12, -2)
    # print(mask)

    # print("\nShould be identical:............")
    # for i in range(3):
    #     print("-")
    #     mask = masker.make_mask(batch, 4, -2)
    #     print(mask)

    # print("\nShould not have changed............")
    # masker = MaskMakerSingleton()
    # for i in range(3):
    #     print("-")
    #     mask = masker.make_mask(batch, 4, -2)
    #     print(mask)

    # masker.reseed()

    # print("\nShould be different from above:............")
    # for i in range(3):
    #     print("-")
    #     mask = masker.make_mask(batch, 4, -2)
    #     print(mask)

    # print("\nAdditional Dimension:............")
    # batch = torch.ones((5,3,4))
    # for i in range(3):
    #     print("-")
    #     mask = masker.make_mask(batch, 4, -2, axis=-1)
    #     print(mask)