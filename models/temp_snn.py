"""
This code is from https://github.com/Horizon2333/imagenet-autoencoder

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from torch.autograd import Variable


class Purifier_Classifier(nn.Module):
    def __init__(self, purifier, classifier):
        super(Purifier_Classifier, self).__init__()
        self.purifier = purifier
        self.classifier = classifier
        
    def forward(self, x, return_image=False):
        
        x_tilde, loss = self.purifier(x)
        y_pred = self.classifier(x_tilde)
        
        if return_image:
            return y_pred, x_tilde, loss
        else:
            return y_pred
        

class Stochastic(nn.Module):
    def __init__(self, in_feature, out_feature):
        super(Stochastic, self).__init__()
        """ Reparameterization trick 시, channel dimension만 계산하는 것이 맞을까? 어떻게 해야할까 """
        
        self.in_feature = in_feature
        self.out_feature = out_feature
        
        self.mu = nn.Linear(in_feature, out_feature)
        self.log_var = nn.Linear(in_feature, out_feature)
    
    def forward(self, input):
        """ If input shape is [B, C, H, W] """
        if len(input.shape) == 4: 
            input = input.transpose(0, 2, 3, 1) # [B, C, H, W] -> [B, H, W, C]
    
        mu = self.mu(input)
        log_var = F.softplus(self.log_var(input))
        
        z = self.reparameterize(mu, log_var)
        
        if len(input.shape) == 4:
            z = z.transpose(0, 3, 1, 2) # [B, H, W, C] -> [B, C, H, W]
            
        return z, mu, log_var
    
    def reparameterize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)
        
        if mu.is_cuda:
            epsilon = epsilon.cuda()
        
        std = log_var.mul(0.5).exp_()
        
        z = mu.addcmul(std, epsilon)
        
        return z
    
    
class PAP(nn.Module):
    def __init__(self, num_layers=4, norm_layer=nn.BatchNorm2d, activation=nn.LeakyReLU):
        super(PAP, self).__init__()
        
        first_conv = [3, 1, 1]
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=first_conv[0], stride=first_conv[1], padding=first_conv[2], bias=False),
            norm_layer(num_features=64),
            activation(inplace=True),
        )
        
        num_layers = [3, 4, 6, 3]
        # Encoder
        self.enc_block1 = EncoderResidualBlock(in_channels=64,  hidden_channels=128,  layers=num_layers[0], downsample_method="conv",
                                               norm_layer=norm_layer, activation=activation)
        self.enc_block2= EncoderResidualBlock(in_channels=128,  hidden_channels=256, layers=num_layers[1], downsample_method="conv",
                                              norm_layer=norm_layer, activation=activation)
        self.enc_block3 = EncoderResidualBlock(in_channels=256,  hidden_channels=512,  layers=num_layers[2], downsample_method="conv",
                                               norm_layer=norm_layer, activation=activation)
        self.enc_block4= EncoderResidualBlock(in_channels=512,  hidden_channels=512, layers=num_layers[3], downsample_method="conv",
                                              norm_layer=norm_layer, activation=activation)
        
        self.dec_block1 = DecoderResidualBlock(hidden_channels=512, output_channels=512,  layers=num_layers[3],
                                               norm_layer=norm_layer, activation=activation)
        self.dec_block2 = DecoderResidualBlock(hidden_channels=512, output_channels=256,  layers=num_layers[2],
                                               norm_layer=norm_layer, activation=activation)
        self.dec_block3 = DecoderResidualBlock(hidden_channels=256, output_channels=128,  layers=num_layers[1],
                                               norm_layer=norm_layer, activation=activation)
        self.dec_block4 = DecoderResidualBlock(hidden_channels=128, output_channels=64,  layers=num_layers[0],
                                               norm_layer=norm_layer, activation=activation)
        
        last_conv = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=first_conv[0], stride=first_conv[1], padding=first_conv[2], bias=False)
        self.dec_conv = nn.Sequential(
            norm_layer(num_features=64),
            activation(inplace=True),
            last_conv,
        )
        self.gate = nn.Sigmoid()
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.activation = activation()
        self.latent_stochastic1 = Stochastic(512, 512)
        self.latent_stochastic2 = Stochastic(512, 512 * 4)
    
    def forward(self, x):
        # Encoder
        x = self.enc_conv(x)
        res = x
        x = self.enc_block1(x)
        res1 = x
        x = self.enc_block2(x)
        res2 = x
        x = self.enc_block3(x)
        res3 = x
        x = self.enc_block4(x)
        
        x = self.avgpool(x).squeeze(-1).squeeze(-1) # (B, D, 1, 1) -> (B, D)
        x, _, log_var1 = self.latent_stochastic1(x)
        x = self.activation(x)
        x, _, log_var2 = self.latent_stochastic2(x)
        x = self.activation(x)
        
        margin = 1
        # print(f"log_var1: {log_var1.mean().item()}")
        # print(f"log_var2: {log_var2.mean().item()}")
        assert not log_var1.isnan().any()
        assert not log_var2.isnan().any()
        loss = (self.get_loss(log_var1, margin) + self.get_loss(log_var2, margin)) / 2.0
        
        x = x.reshape(x.size(0), 512, 2, 2)
        
        # Decoder
        x = self.dec_block1(x)
        x += res3
        x = self.dec_block2(x)
        x += res2
        x = self.dec_block3(x)
        x += res1
        x = self.dec_block4(x)
        x += res
        x = self.dec_conv(x)
        # Sigmoid
        x = self.gate(x)
        
        return x, loss
    
    def get_loss(self, log_var, margin):
        return torch.max(margin - log_var, 0)[0].mean()
    
# =============================================================================

class EncoderResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, layers, downsample_method="conv",
                 norm_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU):
        super(EncoderResidualBlock, self).__init__()

        if downsample_method == "conv":
            for i in range(layers):
                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, downsample=True,
                                                 norm_layer=norm_layer, activation=activation)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, downsample=False,
                                                 norm_layer=norm_layer, activation=activation)
                self.add_module('%02d EncoderLayer' % i, layer)
        
        elif downsample_method == "pool":
            maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.add_module('00 MaxPooling', maxpool)
            for i in range(layers):
                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, downsample=False,
                                                 norm_layer=norm_layer, activation=activation)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, downsample=False,
                                                 norm_layer=norm_layer, activation=activation)
                self.add_module('%02d EncoderLayer' % (i+1), layer)
        else:
            for i in range(layers):
                if i == 0:
                    layer = EncoderResidualLayer(in_channels=in_channels, hidden_channels=hidden_channels, downsample=False,
                                                 norm_layer=norm_layer, activation=activation)
                else:
                    layer = EncoderResidualLayer(in_channels=hidden_channels, hidden_channels=hidden_channels, downsample=False,
                                                 norm_layer=norm_layer, activation=activation)
                self.add_module('%02d EncoderLayer' % (i+1), layer)
    
    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)
        return x
    
class DecoderResidualBlock(nn.Module):
    def __init__(self, hidden_channels, output_channels, layers,
                 norm_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU):
        super(DecoderResidualBlock, self).__init__()

        for i in range(layers):
            if i == layers - 1:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=output_channels, upsample=True,
                                             norm_layer=norm_layer, activation=activation)
            else:
                layer = DecoderResidualLayer(hidden_channels=hidden_channels, output_channels=hidden_channels, upsample=False,
                                             norm_layer=norm_layer, activation=activation)
            self.add_module('%02d EncoderLayer' % i, layer)
    
    def forward(self, x):
        for name, layer in self.named_children():
            x = layer(x)
        return x
    

class EncoderResidualLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, downsample,
                 norm_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU):
        super(EncoderResidualLayer, self).__init__()

        if downsample:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=2, padding=1, bias=False),
                norm_layer(num_features=hidden_channels),
                activation(inplace=True),
            )
        else:
            self.weight_layer1 = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
                norm_layer(num_features=hidden_channels),
                activation(inplace=True),
            )

        self.weight_layer2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            norm_layer(num_features=hidden_channels),
        )

        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1, stride=2, padding=0, bias=False),
                norm_layer(num_features=hidden_channels),
            )
        else:
            self.downsample = None

        self.activation = nn.Sequential(
            activation(inplace=True)
        )
    
    def forward(self, x):
        identity = x
        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = x + identity
        x = self.activation(x)
        return x
    

class DecoderResidualLayer(nn.Module):
    def __init__(self, hidden_channels, output_channels, upsample,
                 norm_layer=nn.InstanceNorm2d, activation=nn.LeakyReLU):
        super(DecoderResidualLayer, self).__init__()

        self.weight_layer1 = nn.Sequential(
            norm_layer(num_features=hidden_channels),
            activation(inplace=True),
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
        )

        if upsample:
            self.weight_layer2 = nn.Sequential(
                norm_layer(num_features=hidden_channels),
                activation(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)                
            )
        else:
            self.weight_layer2 = nn.Sequential(
                norm_layer(num_features=hidden_channels),
                activation(inplace=True),
                nn.Conv2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1, bias=False),
            )

        if upsample:
            self.upsample = nn.Sequential(
                norm_layer(num_features=hidden_channels),
                activation(inplace=True),
                nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=output_channels, kernel_size=1, stride=2, output_padding=1, bias=False)   
            )
        else:
            self.upsample = None
    
    def forward(self, x):
        identity = x
        x = self.weight_layer1(x)
        x = self.weight_layer2(x)
        if self.upsample is not None:
            identity = self.upsample(identity)
        x = x + identity
        return x
    
