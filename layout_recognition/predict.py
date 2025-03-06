import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from pdf2image import convert_from_path
import sys
import cv2

# from layout_recognition.model_script import get_model
# from layout_recognition.evaluate_script import load_model

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
        
        x = self.bottleneck(x)
        
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


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_pyramid(image, min_size=512, scale_factor=0.75):
    """
    For image resizing

    """
    # Convert PIL to numpy for OpenCV
    img_np = np.array(image)
    
    # Create pyramid
    pyramid = []
    current_img = img_np.copy()
    pyramid.append(Image.fromarray(current_img))
    while True:
        height, width = current_img.shape[:2]
        new_height = int(height * scale_factor)
        new_width = int(width * scale_factor)
        
        if new_height < min_size or new_width < min_size:
            break
        current_img = cv2.resize(current_img, (new_width, new_height), 
                                interpolation=cv2.INTER_AREA)
        pyramid.append(Image.fromarray(current_img))
    
    return pyramid
def load_model(model_path, model_name='cnn_rnn_model', device='cpu'):
    """
    Load a trained model.
    
    """    
    print(f"Loading model from: {model_path}")
    print(f"Model name: {model_name}")

    model = get_model(model_name)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model
def predict_layout(pdf_path, model_path=None, model_name='cnn_rnn_model', output_dir='results/predictions'):

    os.makedirs(output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Converting PDF to image: {pdf_path}")
    images = convert_from_path(pdf_path, dpi=150, poppler_path = r"C:\Users\pwal9\Downloads\Release-24.08.0-0 (1)\poppler-24.08.0\Library\bin")

    if model_path is None:
        model_dir = os.path.join('models', 'layout')
        model_paths = [os.path.join(model_dir, d, 'best_model.pth') for d in os.listdir(model_dir) 
                      if os.path.isdir(os.path.join(model_dir, d))]
        model_paths = [p for p in model_paths if os.path.exists(p)]
        
        if not model_paths:
            print("No trained model found. Please train a model first.")
            return
        
        model_path = sorted(model_paths, key=lambda p: os.path.getmtime(p))[-1]
    
    print(f"Using model: {model_path}")

    model = load_model(model_path, model_name, device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    for i, img in enumerate(images):
        print(f"Processing page {i+1}/{len(images)}")
        pyramid = create_pyramid(img)
        print(f"Created pyramid with {len(pyramid)} levels")

        combined_pred = np.zeros((512, 512), dtype=np.float32)

        for level, pyramid_img in enumerate(pyramid):
            img_tensor = transform(pyramid_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(img_tensor)
            level_pred = output.cpu().squeeze().numpy()
            if level_pred.shape != (512, 512):
                level_pred = cv2.resize(level_pred, (512, 512))
            
            weight = 1.0 / (level + 1)
            combined_pred += level_pred * weight
    
        combined_pred = combined_pred / sum(1.0 / (l + 1) for l in range(len(pyramid)))
        pred_binary = (combined_pred > 0.5).astype(np.float32)
        
        img_np = np.array(img.resize((512, 512)))
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        overlay = img_np.copy()
        red_mask = np.zeros_like(overlay)
        red_mask[:, :, 0] = 255
        
        alpha = 0.3
        mask_bool = pred_binary > 0.5
        
        if mask_bool.shape != overlay.shape[:2]:
            mask_bool_resized = np.zeros(overlay.shape[:2], dtype=bool)
            h, w = min(mask_bool.shape[0], overlay.shape[0]), min(mask_bool.shape[1], overlay.shape[1])
            mask_bool_resized[:h, :w] = mask_bool[:h, :w]
            mask_bool = mask_bool_resized
        
        for c in range(3):
            overlay_channel = overlay[:, :, c]
            red_channel = red_mask[:, :, c]
            overlay_channel[mask_bool] = (1-alpha) * overlay_channel[mask_bool] + alpha * red_channel[mask_bool]
            overlay[:, :, c] = overlay_channel
        
        axes[1].imshow(overlay)
        axes[1].set_title('Predicted Text Regions')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'prediction_page_{i+1}.png'))
        plt.close()
    
    print(f"Predictions saved to {output_dir}")

if __name__ == "__main__":
    pdf_path = os.path.join('..', 'sample.pdf')
    predict_layout(pdf_path)