
from mxnet.gluon import nn


# __all__ = ["params_dict", "MBBlock", "UpSampling", "conv_bn", "conv_1x1_bn", "AdaptiveAvgPool2D"]

params_dict = {
      # (width_coefficient, depth_coefficient, resolution, dropout_rate)
      'b0': (1.0, 1.0, 224, 0.2),
      'b1': (1.0, 1.1, 240, 0.2),
      'b2': (1.1, 1.2, 260, 0.3),
      'b3': (1.2, 1.4, 300, 0.3),
      'b4': (1.4, 1.8, 380, 0.4),
      'b5': (1.6, 2.2, 456, 0.4),
      'b6': (1.8, 2.6, 528, 0.5),
      'b7': (2.0, 3.1, 600, 0.5),
  }

# Swish Activation is gluon.nn.Swish(beta=1.0)


class AdaptiveAvgPool2D(nn.HybridBlock):
    def __init__(self, output_size):
        super(AdaptiveAvgPool2D, self).__init__()
        self.output_size = output_size
    
    def hybrid_forward(self, F, x):
        return F.contrib.AdaptiveAvgPooling2D(x, self.output_size)



class SEModule(nn.HybridBlock):
    def __init__(self, channel, reduction=2):
        super(SEModule, self).__init__()
        # self.avg_pool = nn.contrib.AdaptiveAvgPooling2D()
        self.fc = nn.HybridSequential()
        self.fc.add(nn.Dense(channel//reduction, use_bias=False),
                    nn.Activation("relu"),
                    nn.Dense(channel, use_bias=False),
                    nn.Activation("sigmoid")) # in mobilenet-v3, this is Hsigmoid
    
    def hybrid_forward(self, F, x):
        res = x
        w = F.contrib.AdaptiveAvgPooling2D(x, output_size=1)
        w = self.fc(w)
        x = F.broadcast_mul(x, w.expand_dims(axis=2).expand_dims(axis=2))
        # x = F.Activation(x + res, act_type='relu')
        return x


def conv_bn(channels, kernel_size, stride, groups=1, activation=nn.Activation('relu')):
    out = nn.HybridSequential()
    if groups == 1:
        out.add(
            nn.Conv2D(channels, kernel_size, stride, kernel_size//2, use_bias=False),
            nn.BatchNorm(scale=True),
            activation
        )
    else:
        out.add(
            nn.Conv2D(channels, kernel_size, stride, kernel_size//2, groups=groups, use_bias=False),
            nn.BatchNorm(scale=True),
            activation
        )
    return out


def conv_1x1_bn(channels, groups=1, activation=nn.Activation('relu')):
    out = nn.HybridSequential()
    out.add(
        nn.Conv2D(channels, 1, 1, 0, use_bias=False),
        nn.BatchNorm(scale=True),
        activation
    )
    return out



class BottleNeck(nn.HybridBlock):
    def __init__(self, channel, kernel_size, stride, expand=1.0, se_ratio=1.0, res_add=True):
        super(BottleNeck, self).__init__()
        self.add=res_add
        if expand==1.0:
            self.out = nn.HybridSequential()
            self.out.add(
                conv_bn(channel, kernel_size, stride, groups=channel, activation=nn.Swish()),
                SEModule(channel, se_ratio),
                nn.BatchNorm(scale=False)
            )
        else:
            self.out = nn.HybridSequential()
            self.out.add(
                conv_1x1_bn(channel*expand, activation=nn.Swish()),
                conv_bn(channel*expand, kernel_size, stride, groups=channel*expand, activation=nn.Swish()),
                SEModule(channel*expand, se_ratio),
                conv_1x1_bn(channel, activation=nn.Swish())
            )
    
    def hybrid_forward(self, F, x):
        output = self.out(x)
        return x + output if self.add else output


class MBBlock(nn.HybridBlock):
    def __init__(self, channel, repeat_num, kernel_size, stride, expand, se_ratio):
        super(MBBlock, self).__init__()
        layers=[BottleNeck(channel, kernel_size, stride, expand, se_ratio, False)]
        layers += [BottleNeck(channel, kernel_size, 1, expand, se_ratio) for _ in range(1, repeat_num)]
        self.out = nn.HybridSequential()
        self.out.add(*layers)
    
    def hybrid_forward(self, F, x):
        return self.out(x)


class UpSampling(nn.HybridBlock):
    def __init__(self, scale=1):
        super(UpSampling, self).__init__()
        self.scale = scale
    
    def hybrid_forward(self, F, x):
        return F.UpSampling(x, scale=self.scale, sample_type='bilinear')


# Not Used Blocks
class _AdaptiveAvgPool2D(nn.HybridBlock):
    def __init__(self, output_h, output_w):
        super(_AdaptiveAvgPool2D, self).__init__()
        self.h = output_h
        self.w = output_w
    
    def hybrid_forward(self, x):
        _, _, in_h, in_w = x.shape
        h_s = in_h/self.h   # if in python3, should be in_h//self.h
        w_s = in_w/self.w   # same as h_s
        pool_h = in_h - (self.h - 1) * h_s
        pool_w = in_w - (self.w - 1) * w_s
        return nn.AvgPool2D((pool_h, pool_w), (h_s, w_s))(x)


class ReLU6(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(ReLU6, self).__init__(**kwargs)

    def hybrid_forward(self, F, x):
        return F.clip(x, 0, 6)


class HSwish(nn.HybridBlock):
    def __init__(self):
        super(HSwish, self).__init__()
    
    def hybrid_forward(self, F, x):
        # return x * F.relu6(x + 3., inplace=self.inplace) / 6.
        return x * ReLU6(x+3) / 6.


class HSigmoid(nn.HybridBlock):
    def __init__(self):
        super(HSigmoid, self).__init__()
    
    def hybrid_forward(self, x):
        return ReLU6(x) /6.