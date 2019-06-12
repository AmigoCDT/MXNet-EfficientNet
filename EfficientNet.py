import mxnet as mx
import utils
from mxnet.gluon import nn
# import symbol_utils


class EfficientNet(nn.HybridBlock):
    def __init__(self, width_coeff=1.0, depth_coeff=1.0, dropout_rate=0.0, scale=1, se_ratio=0.25, num_classes=256):
        super(EfficientNet, self).__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        expands = [1, 6, 6, 6, 6, 6, 6]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]

        channels = [round(x*width_coeff) for x in channels] # [int(x*width) for x in channels]
        repeats = [round(x*depth_coeff) for x in repeats] # [int(x*width) for x in repeats]

        self.out = nn.HybridSequential()
        if scale!=1:
            # Here, should do interpolation to resize input_image to resolution in "bi" mode
            # self.out.add(utils.UpSampling(scale))
            pass
        self.out.add(nn.Conv2D(channels[0], 3, 2, padding=1, use_bias=False))
        self.out.add(nn.BatchNorm(scale=True))
        for i in range(7):
            self.out.add(utils.MBBlock(channels[i+1], repeats[i], kernel_sizes[i], strides[i], expands[i], se_ratio))
        self.out.add(utils.conv_1x1_bn(channels[8], nn.Swish()),
                    utils.AdaptiveAvgPool2D(1),
                    nn.Flatten(),
                    nn.Dropout(dropout_rate),
                    nn.Dense(num_classes, use_bias=False),
                    nn.BatchNorm(scale=True),
                    nn.Swish())
                    # utils.conv_1x1_bn(num_classes, nn.Swish()))
        # print(self.out)

    def hybrid_forward(self, F, x):
        feature = self.out(x)
        # print(feature.shape)
        return feature


class FC(nn.HybridBlock):
    def __init__(self, num_classes):
        super(FC, self).__init__()
        self.out = nn.HybridSequential()
        self.out.add(nn.BatchNorm(scale=False),
                    nn.Dropout(0.4),
                    nn.Dense(num_classes, use_bias=False),
                    nn.BatchNorm(scale=False))
    
    def hybrid_forward(self, F, x):
        return self.out(x)


def get_symbol(model_name="b0", num_classes=512):
    print("embedding size: {}".format(num_classes))
    width_coeff, depth_coeff, input_resolution, dropout_rate = utils.params_dict[model_name]
    net = EfficientNet(width_coeff, depth_coeff, dropout_rate, scale=input_resolution/224, num_classes=512)
    data = mx.sym.Variable(name='data')
    # data = (data-127.5)
    data = (data-127.5)*0.0078125
    body = net(data)
    # fc_classes = kwargs.get("fc_classes", 0)
    # if fc_classes > 0:
    #     # import symbol_utils
    #     # body = FC(fc_classes)(body)
    #     body = nn.BatchNorm(scale=False)(body)
    #     body = nn.Dropout(0.4)(body)
    #     body = nn.Dense(num_classes, use_bias=False)(body)
    #     body = nn.BatchNorm(scale=False)(body)
    return body

