"""Part of the code here is from Texar-pytorch

# Copyright 2019 The Texar Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

import torch
from torch import nn
from torch.nn import functional as F

def safe_cumprod(x: torch.Tensor,
                 *args,
                 **kwargs) -> torch.Tensor:
    r"""Computes cumprod of x in logspace using cumsum to avoid underflow.
    The cumprod function and its gradient can result in numerical
    instabilities when its argument has very small and/or zero values.
    As long as the argument is all positive, we can instead compute the
    cumulative product as `exp(cumsum(log(x)))`.  This function can be called
    identically to :torch:`cumprod`.
    Args:
        x: Tensor to take the cumulative product of.
        *args: Passed on to cumsum; these are identical to those in cumprod.
        **kwargs: Passed on to cumsum; these are identical to those in cumprod.
    Returns:
        Cumulative product of x.
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)

    tiny = torch.finfo(x.dtype).tiny

    return torch.exp(torch.cumsum(torch.log(torch.clamp(x, tiny, 1)),
                                  *args, **kwargs))

def monotonic_attention(p_choose_i: torch.Tensor,
                        previous_attention: torch.Tensor,
                        mode: str) -> torch.Tensor:
    r"""Compute monotonic attention distribution from choosing probabilities.
    Monotonic attention implies that the input sequence is processed in an
    explicitly left-to-right manner when generating the output sequence.  In
    addition, once an input sequence element is attended to at a given output
    time step, elements occurring before it cannot be attended to at subsequent
    output time steps.  This function generates attention distributions
    according to these assumptions. For more information, see `Online and
    Linear-Time Attention by Enforcing Monotonic Alignments`.
    Args:
        p_choose_i: Probability of choosing input sequence/memory element i.
            Should be of shape (batch_size, input_sequence_length), and should
            all be in the range [0, 1].
        previous_attention: The attention distribution from the previous output
            time step.  Should be of shape (batch_size, input_sequence_length).
            For the first output time step, `previous_attention[n]` should be
            `[1, 0, 0, ..., 0] for all n in [0, ... batch_size - 1]`.
        mode: How to compute the attention distribution.
            Must be one of ``"recursive"``, ``"parallel"``, or ``"hard"``:
            - ``"recursive"`` recursively computes the distribution.
              This is slowest but is exact, general, and does not suffer
              from numerical instabilities.
            - ``"parallel"`` uses parallelized cumulative-sum and
              cumulative-product operations to compute a closed-form
              solution to the recurrence relation defining the attention
              distribution. This makes it more efficient than
              ``"recursive"``, but it requires numerical checks which make
              the distribution non-exact. This can be a problem in
              particular when input sequence is long and/or
              :attr:`p_choose_i` has entries very close to 0 or 1.
            - ``"hard"`` requires that the probabilities in
              :attr:`p_choose_i` are all either 0 or 1, and subsequently
              uses a more efficient and exact solution.
    Returns:
        A tensor of shape (batch_size, input_sequence_length) representing the
        attention distributions for each sequence in the batch.
    Raises:
        ValueError: mode is not one of ``"recursive"``, ``"parallel"``,
            ``"hard"``.
    """
    # Force things to be tensors
    if not isinstance(p_choose_i, torch.Tensor):
        p_choose_i = torch.tensor(p_choose_i)

    if not isinstance(previous_attention, torch.Tensor):
        previous_attention = torch.tensor(previous_attention)

    if mode == "recursive":
        # Use .shape[0] when it's not None, or fall back on symbolic shape
        batch_size = p_choose_i.shape[0]
        # Compute [1, 1 - p_choose_i[0], 1 - p_choose_i[1], ...,
        # 1 - p_choose_i[-2]]
        shifted_1mp_choose_i = torch.cat((p_choose_i.new_ones(batch_size, 1),
                                          1 - p_choose_i[:, :-1]), 1)

        # Compute attention distribution recursively as
        # q[i] = (1 - p_choose_i[i - 1])*q[i - 1] + previous_attention[i]
        # attention[i] = p_choose_i[i]*q[i]

        def f(x, yz):
            return torch.reshape(yz[0] * x + yz[1], (batch_size,))

        x_tmp = f(torch.zeros((batch_size,)), torch.transpose(
            shifted_1mp_choose_i, 0, 1))
        x_tmp = f(x_tmp, torch.transpose(previous_attention, 0, 1))
        attention = p_choose_i * torch.transpose(x_tmp, 0, 1)
    elif mode == "parallel":
        batch_size = p_choose_i.shape[0]
        shifted_1mp_choose_i = torch.cat((p_choose_i.new_ones(batch_size, 1),
                                          1 - p_choose_i[:, :-1]), 1)
        # safe_cumprod computes cumprod in logspace with numeric checks
        cumprod_1mp_choose_i = safe_cumprod(shifted_1mp_choose_i, dim=1)
        # Compute recurrence relation solution
        attention = p_choose_i * cumprod_1mp_choose_i * torch.cumsum(
            previous_attention / cumprod_1mp_choose_i.clamp(min=1e-10, max=1.),
            dim=1)
    elif mode == "hard":
        # Remove any probabilities before the index chosen last time step
        p_choose_i *= torch.cumsum(previous_attention, dim=1)
        # Now, use exclusive cumprod to remove probabilities after the first
        # chosen index, like so:
        # p_choose_i = [0, 0, 0, 1, 1, 0, 1, 1]
        # cumprod(1 - p_choose_i, exclusive=True) = [1, 1, 1, 1, 0, 0, 0, 0]
        # Product of above: [0, 0, 0, 1, 0, 0, 0, 0]
        batch_size = p_choose_i.shape[0]
        shifted_1mp_choose_i = torch.cat((p_choose_i.new_ones(batch_size, 1),
                                          1 - p_choose_i[:, :-1]), 1)
        attention = p_choose_i * torch.cumprod(shifted_1mp_choose_i, dim=1)
    else:
        raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")
    return attention

def _monotonic_probability_fn(score: torch.Tensor,
                              previous_alignments: torch.Tensor,
                              sigmoid_noise: float,
                              mode: str) -> torch.Tensor:
    r"""Attention probability function for monotonic attention.
    Takes in unnormalized attention scores, adds pre-sigmoid noise to
    encourage the model to make discrete attention decisions, passes them
    through a sigmoid to obtain "choosing" probabilities, and then calls
    monotonic_attention to obtain the attention distribution.  For more
    information, see
    `Colin Raffel, Minh-Thang Luong, Peter J. Liu, Ron J. Weiss, Douglas Eck,
    "Online and Linear-Time Attention by Enforcing Monotonic Alignments."
    ICML 2017.  https://arxiv.org/abs/1704.00784`_
    Args:
        score: Unnormalized attention scores, shape
            ``[batch_size, alignments_size]``
        previous_alignments: Previous attention distribution, shape
            ``[batch_size, alignments_size]``
        sigmoid_noise: Standard deviation of pre-sigmoid noise.  Setting this
            larger than 0 will encourage the model to produce large attention
            scores, effectively making the choosing probabilities discrete and
            the resulting attention distribution one-hot.  It should be set to 0
            at test-time, and when hard attention is not desired.
        mode: How to compute the attention distribution.  Must be one of
            ``"recursive"``, ``"parallel"``, or ``"hard"``. Refer to
            :func:`~texar.torch.core.monotonic_attention` for more information.
    Returns:
        A ``[batch_size, alignments_size]`` shaped tensor corresponding to the
        resulting attention distribution.
    """
    # Optionally add pre-sigmoid noise to the scores
    if sigmoid_noise > 0:
        noise = torch.randn(score.shape, dtype=score.dtype, device=score.device)
        score += sigmoid_noise * noise
    # Compute "choosing" probabilities from the attention scores
    if mode == "hard":
        # When mode is hard, use a hard sigmoid
        p_choose_i = (score > 0).type(score.dtype)
    else:
        p_choose_i = torch.sigmoid(score)
    # Convert from choosing probabilities to attention distribution
    return monotonic_attention(p_choose_i, previous_alignments, mode)


class LuongAttention(nn.Module):
    """Luong attention mechanisms

    Adapted from: https://pytorch.org/tutorials/beginner/deploy_seq2seq_hybrid_frontend_tutorial.html 
    """
    def __init__(self, method, hidden_size, monotonic = False, monotonic_mode = "recursive", sigmoid_noise = 0.0):
        super(LuongAttention, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

        self.monotonic = monotonic
        self.monotonic_mode = monotonic_mode
        if self.monotonic_mode not in ['recursive', 'parallel', 'hard']:
            raise ValueError(self.monotonic_mode, "is not an appropiate monotonic mode method.")
        self.sigmoid_noise = sigmoid_noise

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs, memory = None):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        if self.monotonic:
            if memory is None:
                raise ValueError("Memory cannot be None with monotonic attention")
            # Monotonic attention
            return _monotonic_probability_fn(attn_energies, memory, self.sigmoid_noise, self.mode).unsqueeze(1)
        else:
            # Normal attention
            # Return the softmax normalized probability scores (with added dimension)
            return F.softmax(attn_energies, dim=1).unsqueeze(1)


