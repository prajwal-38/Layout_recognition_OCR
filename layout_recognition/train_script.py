from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import torch.optim as optim
import torch.nn as nn
from PIL import Image
import numpy as np
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from layout_recognition.model_script import get_model


class LayoutDataset(Dataset):
    """
    Dataset for layout recognition.
    """
    def __init__(self, data_items, transform=None):
        self.data_items = data_items
        self.transform = transform
    
    def __len__(self):
        return len(self.data_items)
    
    def __getitem__(self, idx):
        image_path = self.data_items[idx]['image_path']
        mask_path = self.data_items[idx]['mask_path']
        
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        #transformations
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask)
        else:
            image = transforms.ToTensor()(image)
            mask = transforms.ToTensor()(mask)
        
        #Binarizing mask
        mask = (mask > 0.5).float()
        
        return {'image': image, 'mask': mask}

def dice_loss(pred, target, smooth=1.0):
    """
    Dice loss for layout segmentation.
    
    """
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    
    return 1 - dice.mean()

def dice_coefficient(pred, target, threshold=0.5, smooth=1.0):
    """
    Dice coefficient for evaluation.
    
    """
    pred = (pred > threshold).float()
    
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)
    
    return dice.mean()

def iou_score(pred, target, threshold=0.5, smooth=1.0):
    """
    IoU score for evaluation.

    """
    pred = (pred > threshold).float()
    
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    
    return iou.mean()

def train_model(model, train_loader, val_loader, device, num_epochs=50, learning_rate=1e-4, save_dir='models/layout'):
    """
    Train the layout recognition model.
    
    """
    os.makedirs(save_dir, exist_ok=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    bce_loss = nn.BCELoss()

    history = {
        'train_loss': [],
        'train_dice': [],
        'train_iou': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': []
    }
    
    best_val_dice = 0.0
    
    for epoch in range(num_epochs):
        #Training
        model.train()
        train_loss = 0.0
        train_dice = 0.0
        train_iou = 0.0
        
        for batch in train_loader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(images)
            loss = 0.5 * bce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_dice += dice_coefficient(outputs, masks).item()
            train_iou += iou_score(outputs, masks).item()
        
        train_loss /= len(train_loader)
        train_dice /= len(train_loader)
        train_iou /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                
                loss = 0.5 * bce_loss(outputs, masks) + 0.5 * dice_loss(outputs, masks)
                
                val_loss += loss.item()
                val_dice += dice_coefficient(outputs, masks).item()
                val_iou += iou_score(outputs, masks).item()
        
        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        
        #Update history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"Saved best model with Dice coefficient: {val_dice:.4f}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_dice': val_dice,
            'val_iou': val_iou
        }, os.path.join(save_dir, 'latest_model.pth'))
    
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        json.dump(history, f)
    
    plot_training_history(history, save_dir)
    
    return history

def plot_training_history(history, save_dir):
    """
    Plot training history.
    
    """
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(history['train_dice'], label='Train Dice')
    plt.plot(history['val_dice'], label='Val Dice')
    plt.title('Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(history['train_iou'], label='Train IoU')
    plt.plot(history['val_iou'], label='Val IoU')
    plt.title('IoU Score')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

def load_data_splits(prepared_data_dir):
    """
    Load data splits from JSON file.
    
    """
    with open(os.path.join(prepared_data_dir, 'splits.json'), 'r') as f:
        splits = json.load(f)
    
    #Converting image paths to data items with mask paths :
    train_data = []
    val_data = []
    test_data = []
    
    for image_path in splits['train']:
        mask_path = image_path.replace('_image.png', '_mask.png')
        train_data.append({'image_path': image_path, 'mask_path': mask_path})
    
    for image_path in splits['val']:
        mask_path = image_path.replace('_image.png', '_mask.png')
        val_data.append({'image_path': image_path, 'mask_path': mask_path})
    
    for image_path in splits['test']:
        mask_path = image_path.replace('_image.png', '_mask.png')
        test_data.append({'image_path': image_path, 'mask_path': mask_path})
    
    return {'train': train_data, 'val': val_data, 'test': test_data}

def main():
    torch.manual_seed(42)
    #nvidia-smi
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    #parameters
    batch_size = 16
    num_epochs = 50
    learning_rate = 1e-3
    model_name = 'CNN_RNN_Model'  #'unet' or 'resnet_unet' or 'cnn_rnn_model
    image_size = (512, 512)
    
    #Create timestamp for run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join('models', 'layout', f"{model_name}_{timestamp}")
    
    #Load data
    prepared_data_dir = os.path.join('data', 'prepared')
    splits = load_data_splits(prepared_data_dir)
    
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = LayoutDataset(splits['train'], transform=transform)
    val_dataset = LayoutDataset(splits['val'], transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    model = get_model(model_name, in_channels=3, out_channels=1, pretrained=True)
    model = model.to(device)

    print(f"Model: {model_name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    #Train 
    history = train_model(
        model, 
        train_loader, 
        val_loader, 
        device, 
        num_epochs=num_epochs, 
        learning_rate=learning_rate,
        save_dir=save_dir
    )
    
    print("Training completed!")
    print(f"Best validation Dice coefficient: {max(history['val_dice']):.4f}")
    print(f"Best validation IoU score: {max(history['val_iou']):.4f}")

if __name__ == "__main__":
    main()
