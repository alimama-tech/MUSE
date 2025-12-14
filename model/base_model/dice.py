import torch
from torch import Tensor

def moments(value: torch.Tensor):
    mean = torch.mean(value, axis=0, keepdim=True)
    variance = (torch.detach(mean) - value).pow(2).mean(axis=0, keepdim=True)
    return mean, variance

def batch_normalization(input, mean, variance, offset, scale, epsilon):
    inv = torch.rsqrt(variance + epsilon)
    return (input - mean) * inv * scale + offset

class BatchNorm(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.register_buffer(
            "running_mean", torch.zeros([num_features]), persistent=True
        )
        self.register_buffer("running_var", torch.zeros([num_features]), persistent=True)
        self.running_mean: torch.Tensor
        self.running_var: torch.Tensor
        if affine:
            self.weight = torch.nn.Parameter(torch.ones([num_features]))
            self.bias = torch.nn.Parameter(torch.zeros([num_features]))
        else:
            # self.register_buffer('weight', torch.ones([num_features]), persistent=False)
            # self.register_buffer('bias', torch.zeros([num_features]), persistent=False)
            self.weight = 1.0
            self.bias = 0.0

    def reset_parameters(self,):
        pass

    @torch.no_grad()
    def _assign_update(self, var: torch.Tensor, value: torch.Tensor):
        decay = 1.0 - self.momentum
        update_delta = (var - value) * decay
        var.sub_(update_delta.view(-1))

    def forward(self, input: Tensor, training: bool) -> Tensor:
        training = self.training and training
        input_shape = input.shape
        reshape_input = input.view(-1, input.shape[-1])
        if training:
            mean, variance = moments(reshape_input)
            self._assign_update(self.running_mean, mean)
            self._assign_update(self.running_var, variance)
        else:
            mean, variance = self.running_mean, self.running_var
        return batch_normalization(
            reshape_input, mean, variance, self.bias, self.weight, self.eps
        ).view(input_shape)
    
def bn(
    features_in: int, eps: float = 1e-3, momentum: float = 0.9, affine: bool = True
) -> BatchNorm:
    return BatchNorm(
        num_features=features_in, eps=eps, momentum=momentum, affine=affine
    )

def ln(normalized_shape, elementwise_affine=True, bias=True) -> torch.nn.LayerNorm:
    return torch.nn.LayerNorm(
        normalized_shape=normalized_shape,
        elementwise_affine=elementwise_affine,
        bias=bias,
        eps=1e-12,
    )    

class Dice(torch.nn.Module):
    def __init__(self, feature_in, limit=0, ln_dice=False) -> None:
        super().__init__()
        self.ln_dice = ln_dice
        if ln_dice:
            self.bn = ln(feature_in, False, False)
        else:
            self.bn = BatchNorm(
                num_features=feature_in, eps=1e-4, momentum=0.99, affine=False
            )
        self.limit = limit
        self.dice_gamma = torch.nn.Parameter(torch.empty([1, feature_in]))
        self.reset_parameters()

    def forward(self, param):
        if self.ln_dice:
            out = self.bn(param)
        else:
            out = self.bn(param, param.shape[0] > self.limit)
        logits = torch.nn.functional.sigmoid(out)
        return torch.multiply(self.dice_gamma, (1.0 - logits) * param) + logits * param

    def reset_parameters(self):
        torch.nn.init.constant_(self.dice_gamma, -0.25)

def activation_layer(act_type, units, limit, ln_dice=False):
    if act_type == "sigmoid":
        return torch.nn.Sigmoid()
    elif act_type == "relu":
        return torch.nn.ReLU()
    elif act_type == "prelu":
        return torch.nn.PReLU(num_parameters=units, init=-0.25)
    elif act_type == "dice":
        return Dice(units, limit, ln_dice=ln_dice)
    elif not act_type:
        return torch.nn.Identity()
    else:
        raise RuntimeError("act_type not supprted %s" % (act_type))
