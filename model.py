import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from torchinfo import summary

import math


torch.manual_seed(0)
torch.cuda.manual_seed(0)


class Residual(nn.Module):
  def __init__(self, in_channels, out_channels, num_modules):
      super().__init__()

      self.num_modules = num_modules
      self.pre_conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
        nn.GroupNorm(32, out_channels),
        nn.GELU())

      self.layer_modules = nn.ModuleList()
      for i in range(num_modules):
        if i != num_modules-1:
          module = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, groups=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU())
        else:
          module = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False, groups=1),
            nn.GroupNorm(32, out_channels),
            nn.GELU())

        self.layer_modules.append(module)

  def forward(self, x):
    x = self.pre_conv(x)
    res = x
    for i, m in enumerate(self.layer_modules):
      if i != self.num_modules-1:
        x = m(x) + res
      else:
        x = F.max_pool2d((m(x) + res), 2, 2)

    return x


class ResidualVGG(nn.Module):
  def __init__(self, layer_properties):
      super().__init__()

      self.lp = layer_properties

      self.layers = nn.ModuleList()
      for p in self.lp:
        layer = Residual(*p)
        self.layers.append(layer)
      
      self.coord = nn.Sequential(
        nn.Conv2d(self.lp[-1][-2], 128, 3, 1, 1, bias=False),
        nn.GroupNorm(32, 128),
        nn.GELU(),
        nn.Conv2d(128, 4, 1, 1, 0, bias=True),
      )
      self.fc = nn.Linear(49, 2)

      self.apply(self._init_weights)


  def _init_weights(self, m):
      if isinstance(m, nn.Linear):
          trunc_normal_(m.weight, std=.02)
          if isinstance(m, nn.Linear) and m.bias is not None:
              nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.BatchNorm2d):
          nn.init.constant_(m.bias, 0)
          nn.init.constant_(m.weight, 1.0)
      elif isinstance(m, nn.Conv2d):
          fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          fan_out //= m.groups
          m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
          if m.bias is not None:
              m.bias.data.zero_()


  def forward(self, x):
    for layer in self.layers:
      x = layer(x)

    x = self.coord(x)
    x = x.view(x.size(0), x.size(1), -1)
    x = self.fc(x)

    return x


if __name__ == "__main__":
  res_vgg_layer = [
    [3, 64, 2],  # in_channels, out_channels, num_modules
    [64, 128, 2],  # in_channels, out_channels, num_modules
    [128, 256, 3],  # in_channels, out_channels, num_modules
    [256, 512, 3],  # in_channels, out_channels, num_modules
    [512, 512, 3],  # in_channels, out_channels, num_modules
  ]
  model = ResidualVGG(layer_properties=res_vgg_layer)

  summary(model, input_size=(1, 3, 224, 224), device='cpu')

  x = torch.randn(1, 3, 224, 224)

  y = model(x)