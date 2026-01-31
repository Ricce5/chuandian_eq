import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str, mask: torch.Tensor = None):
        """Apply RevIN normalization or its inverse.

        Args:
            x: Tensor of shape (B, ..., C) where C is the feature dimension (last dim).
            mode: 'norm' to normalize, 'denorm' to revert.
            mask: Optional tensor broadcastable to x without the last dimension,
                  typically shape (B, L) or (B, L, 1). Ones indicate valid (non-pad).
        """
        if mode == 'norm':
            self._get_statistics(x, mask)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor, mask: torch.Tensor = None):
        # reduce over all dims except batch and last feature dim
        dim2reduce = tuple(range(1, x.ndim - 1))
        if mask is None:
            mean = torch.mean(x, dim=dim2reduce, keepdim=True)
            var = torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False)
        else:
            # Ensure mask shape is broadcastable to x without feature dim
            if mask.dtype != torch.float32 and mask.dtype != torch.float64:
                mask = mask.float()
            # Expand mask to cover feature dim for broadcasting
            while mask.ndim < x.ndim - 1:
                mask = mask.unsqueeze(-1)
            # Broadcast mask across feature dim if needed
            if mask.ndim == x.ndim - 1:
                mask_exp = mask.unsqueeze(-1)
            else:
                mask_exp = mask

            # Use expanded mask to ensure correct broadcasting and reduced shape
            denom = mask_exp.sum(dim=dim2reduce, keepdim=True).clamp_min(1.0)
            mean = (x * mask_exp).sum(dim=dim2reduce, keepdim=True) / denom
            var = ((x - mean) ** 2 * mask_exp).sum(dim=dim2reduce, keepdim=True) / denom

        stdev = torch.sqrt(var + self.eps)
        # Detach to avoid backprop through statistics as in original RevIN
        self.mean = mean.detach()
        self.stdev = stdev.detach()

    def _normalize(self, x: torch.Tensor):
        x = (x - self.mean) / self.stdev
        if self.affine:
            w = self._affine_weight_broadcast(x)
            b = self._affine_bias_broadcast(x)
            x = x * w + b
        return x

    def _denormalize(self, x: torch.Tensor):
        if self.affine:
            w = self._affine_weight_broadcast(x)
            b = self._affine_bias_broadcast(x)
            x = (x - b) / w.clamp_min(self.eps)
        x = x * self.stdev + self.mean
        return x

    def _affine_weight_broadcast(self, x: torch.Tensor) -> torch.Tensor:
        # reshape (C,) -> (1, ..., 1, C) to match x
        shape = [1] * (x.ndim - 1) + [self.num_features]
        return self.affine_weight.view(*shape)

    def _affine_bias_broadcast(self, x: torch.Tensor) -> torch.Tensor:
        shape = [1] * (x.ndim - 1) + [self.num_features]
        return self.affine_bias.view(*shape)
