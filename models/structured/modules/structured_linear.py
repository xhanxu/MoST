import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class StructuredLinear(nn.Module):

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self._set_extended_features()
        self._initialize_bias(bias, factory_kwargs)

    def _set_extended_features(self):
        if not hasattr(self, 'in_features_extended'):
            self.in_features_extended = self.in_features
        if not hasattr(self, 'out_features_extended'):
            self.out_features_extended = self.out_features

    def _initialize_bias(self, bias, factory_kwargs):
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

    def reset_parameters(self) -> None:
        self._reset_weights_with_kaiming()
        self.reset_parameters_bias()

    def _reset_weights_with_kaiming(self):
        dense_init_fn = partial(init.kaiming_uniform_, a=math.sqrt(5))
        self.set_weights_from_dense_init(dense_init_fn_=dense_init_fn)

    def set_weights_from_dense_init(self, dense_init_fn_):
        raise NotImplementedError

    def reset_parameters_bias(self):
        if self.bias is not None:
            self._apply_uniform_bias_init()

    def _apply_uniform_bias_init(self):
        fan_in = self.bias.shape[-1]
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)

    @property
    def saving(self):
        raise NotImplementedError

    def convert_to_dense_weight(self):
        factory_kwargs = self._get_weight_factory_kwargs()
        identity_matrix = torch.eye(self.in_features, **factory_kwargs)
        dense_weight = self.forward_matmul(identity_matrix).T
        return dense_weight

    def _get_weight_factory_kwargs(self):
        return {'device': self.weight.device, 'dtype': self.weight.dtype}

    def preprocess(self, x):
        return self._pad_input_if_needed(x)

    def _pad_input_if_needed(self, x):
        in_features = x.shape[-1]
        if in_features < self.in_features_extended:
            padding = self.in_features_extended - in_features
            x = F.pad(x, (0, padding))
        return x

    def postprocess(self, output):
        return self._truncate_output_if_needed(output)

    def _truncate_output_if_needed(self, output):
        out_features_extended = output.shape[-1]
        if out_features_extended > self.out_features:
            output = output[..., :self.out_features]
        return output

    def forward_matmul(self, x):
        raise NotImplementedError

    def forward(self, x):
        output = self.forward_matmul(x)
        return self._add_bias_if_present(output)

    def _add_bias_if_present(self, output):
        if self.bias is not None:
            bias_corrected = self.bias.to(dtype=output.dtype)
            return output + bias_corrected
        return output
