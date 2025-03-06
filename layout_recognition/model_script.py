import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvBlock(nn.Module):
    """
    Convolutional block with batch normalization and ReLU activation.
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        #self.dropout = nn.Dropout(p=dropout_rate)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        #x = self.dropout(x)
        return x

class EncoderBlock(nn.Module):
    """
    Encoder block for U-Net.
    """
    def __init__(self, in_channels, out_channels):
        super(EncoderBlock, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        features = self.conv1(x)
        features = self.conv2(features)
        pooled = self.pool(features)
        return pooled, features

class DecoderBlock(nn.Module):
    """
    Decoder block for U-Net.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(out_channels + skip_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x, skip):
        x = self.upconv(x)
        
        diff_h = skip.size()[2] - x.size()[2]
        diff_w = skip.size()[3] - x.size()[3]
        
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UNet(nn.Module):
    """
    U-Net model for layout recognition.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        self.enc1 = EncoderBlock(in_channels, 64)
        self.enc2 = EncoderBlock(64, 128)
        self.enc3 = EncoderBlock(128, 256)
        self.enc4 = EncoderBlock(256, 512)
        
        self.bottleneck = nn.Sequential(
            ConvBlock(512, 1024),
            ConvBlock(1024, 1024)
        )
        
        self.dec4 = DecoderBlock(1024, 512, 512)
        self.dec3 = DecoderBlock(512, 256, 256)
        self.dec2 = DecoderBlock(256, 128, 128)
        self.dec1 = DecoderBlock(128, 64, 64)
        
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        ###
        x = self.bottleneck(x)
        ###
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        x = self.output(x)
        return torch.sigmoid(x)

class ResNetUNet(nn.Module):
    """
    U-Net model with ResNet backbone for layout recognition.
    """
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super(ResNetUNet, self).__init__()
        
        self.encoder = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        
        self.enc1 = nn.Sequential(
            self.encoder.conv1,
            self.encoder.bn1,
            self.encoder.relu
        )
        self.pool = self.encoder.maxpool
        self.enc2 = self.encoder.layer1
        self.enc3 = self.encoder.layer2
        self.enc4 = self.encoder.layer3
        self.enc5 = self.encoder.layer4
        #########
        self.dec5 = DecoderBlock(512, 256, 256) 
        self.dec4 = DecoderBlock(256, 128, 128)
        self.dec3 = DecoderBlock(128, 64, 64)
        self.dec2 = DecoderBlock(64, 64, 64)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            ConvBlock(32, 32),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.pool(enc1)
        enc2 = self.enc2(enc2)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        
        dec5 = self.dec5(enc5, enc4)
        dec4 = self.dec4(dec5, enc3)
        dec3 = self.dec3(dec4, enc2)
        dec2 = self.dec2(dec3, enc1)
        dec1 = self.dec1(dec2)
        
        return torch.sigmoid(dec1)
    
class CNN_RNN_Model(nn.Module):
    """
    CNN-RNN model for layout recognition and segmentation.
    Combines ResNet features with LSTM for sequential processing,
    then upsamples to create full-resolution segmentation maps.
    """
    def __init__(self, in_channels=3, out_channels=1, pretrained=True, hidden_size=256, num_layers=2):
        super(CNN_RNN_Model, self).__init__()
        
        resnet = models.resnet34(weights='IMAGENET1K_V1' if pretrained else None)
        
        self.enc1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.pool = resnet.maxpool  
        self.enc2 = resnet.layer1
        self.enc3 = resnet.layer2
        self.enc4 = resnet.layer3
        self.enc5 = resnet.layer4
        
        self.lstm_hidden_size = hidden_size
        self.lstm = nn.LSTM(512, hidden_size, num_layers, batch_first=True)
        
        self.lstm_proj = nn.Conv2d(hidden_size, 512, kernel_size=1)
        
        self.dec5 = DecoderBlock(512, 256, 256)
        self.dec4 = DecoderBlock(256, 128, 128)
        self.dec3 = DecoderBlock(128, 64, 64)
        self.dec2 = DecoderBlock(64, 64, 64)
        
        self.final_upsampling = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        input_size = x.size()
        batch_size = input_size[0]
        
        enc1 = self.enc1(x)
        enc2 = self.pool(enc1)
        enc2 = self.enc2(enc2)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)         # 1/32 resolution
        
        feature_h, feature_w = enc5.size(2), enc5.size(3)
        features_seq = enc5.view(batch_size, 512, -1).permute(0, 2, 1)
        
        #Run LSTM
        lstm_out, _ = self.lstm(features_seq)
        
        #Reshape back to feature map format
        lstm_out = lstm_out.permute(0, 2, 1).view(batch_size, self.lstm_hidden_size, feature_h, feature_w)
        
        dec_features = self.lstm_proj(lstm_out)
        
        x = self.dec5(dec_features, enc4)
        x = self.dec4(x, enc3)
        x = self.dec3(x, enc2)
        x = self.dec2(x, enc1)
        
        x = self.final_upsampling(x)
        x = self.final_conv(x)
        
        return torch.sigmoid(x)

def get_model(model_name='CNN_RNN_Model', in_channels=3, out_channels=1, pretrained=True):
    """
    Get model by name.
    
    """
    if model_name.lower() == 'unet':
        return UNet(in_channels, out_channels)
    elif model_name.lower() == 'resnet_unet':
        return ResNetUNet(in_channels, out_channels, pretrained)
    elif model_name.lower() == 'cnn_rnn_model':
        return CNN_RNN_Model(in_channels, out_channels, pretrained)
    else:
        raise ValueError(f"Model {model_name} not supported. Choose from 'unet', 'resnet_unet', 'cnn_rnn_model.")
