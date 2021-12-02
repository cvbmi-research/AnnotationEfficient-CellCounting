import torch.nn as nn
import torch, torchvision
from torchvision.models import vgg19, vgg16
from torchsummary import summary
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, in_channels = 3, out_channels = 1, load_weights=False):        
        super(Generator, self).__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat1 = [512, 512, 512, 256, 128, 64]
        self.frontend = make_layers(self.frontend_feat, dilation = 1)
        self.backend1 = make_layers(self.backend_feat1, in_channels = 512, dilation = 2)
        # Attention Mechanism
        self.spatial_attention = make_sp_layers(in_channels=512, out_channels=1, k_size=1)
        self.channel_attention = make_channel_layers(in_channel=512, out_channels=1, reduction_rate=3)
        self.output_layer_mse = nn.Conv2d(64, 1, kernel_size=1) 
        self.relu_g = nn.ReLU(inplace=True)
        self.relu_d = nn.ReLU(inplace=True)
        self.globalMaxPool = nn.AdaptiveMaxPool2d((4,4))
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

        self.upsample_mse = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'), 
                nn.Conv2d(64, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(32, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(32, 64, 3, 1, 1))
              
        if not load_weights:
            mod = vgg16(pretrained = True)
            self._initialize_weights()
            for i in range(len(self.frontend.state_dict().items())):
                list(self.frontend.state_dict().items())[i][1].data[:] = list(mod.state_dict().items())[i][1].data[:]

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.frontend(x)
        x_res = x1
        x_spatial_att = self.spatial_attention(x1)
        x_channel_att = self.channel_attention(x1)
        att_x = x1 * x_spatial_att * x_channel_att.reshape(-1,512,1,1)
        x_back = self.backend1(att_x)
        x_encode = x_back
        x_decode_mse = self.upsample_mse(x_encode)
        out_mse = self.output_layer_mse(x_decode_mse)
        out_mse = self.relu_g(out_mse)
        x3 = self.globalMaxPool(x_encode)
        x3 = x3.view(-1, 1024)
        x3 = F.relu(self.fc1(x3))
        x3 = F.dropout(x3, training=self.training)
        x3 = F.relu(self.fc2(x3))
        x3 = F.dropout(x3, training=self.training)
        out_cluster = F.relu(self.fc3(x3))
        return out_mse, out_cluster

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def make_layers(cfg, in_channels = 3, batch_norm=False, dilation = 1):
    d_rate = dilation
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_sp_layers(in_channels=3, out_channels=1, k_size=1):
    layers = []
    conv2d_1 = nn.Conv2d(in_channels, in_channels//2, kernel_size=k_size, padding=0, dilation=1)
    conv2d_2 = nn.Conv2d(in_channels//2, out_channels, kernel_size=k_size, padding=0, dilation=1)
    layers += [conv2d_1, nn.ReLU(inplace=True), conv2d_2, nn.Sigmoid()]
    return nn.Sequential(*layers)

def make_channel_layers(in_channel=512, out_channels=1, reduction_rate=3): 
    layers = []
    layers += [nn.AdaptiveAvgPool2d(1)]
    conv2d_1 = nn.Conv2d(in_channel, in_channel // reduction_rate, kernel_size=1, padding=0)
    layers += [conv2d_1, nn.ReLU(inplace=True)]
    conv2d_2 = nn.Conv2d(in_channel // reduction_rate, in_channel // reduction_rate, kernel_size=1, padding=0)
    layers += [conv2d_2, nn.ReLU(inplace=True)]
    conv2d_3 = nn.Conv2d(in_channel // reduction_rate, in_channel, kernel_size=1, padding=0)
    layers += [conv2d_3, nn.Softmax(dim=1)]
    return nn.Sequential(*layers)


if __name__ == '__main__':
    net = Generator()
    summary(net, input_size=(3, 256, 256), device='cpu')