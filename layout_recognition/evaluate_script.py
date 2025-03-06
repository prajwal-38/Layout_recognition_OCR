import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from layout_recognition.model_script import get_model
from layout_recognition.train_script import LayoutDataset, dice_coefficient, iou_score

def load_model(model_path, model_name='cnn_rnn_model', device='cpu'):
    """
    Load a trained model.
    
    """    
    print(f"Loading model : {model_path}")
    print(f"Model name: {model_name}")

    model = get_model(model_name)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model

    # model = get_model(model_name)
    # checkpoint = torch.load(model_path, map_location=device)
    # model.load_state_dict(checkpoint['model_state_dict'], strict=False)  # Set strict=False
    # model = model.to(device)
    # model.eval()
    
    # return model

def evaluate_model(model, test_loader, device, threshold=0.5):
    model.eval()
    
    dice_scores = []
    iou_scores = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            outputs = model(images)
            dice = dice_coefficient(outputs, masks, threshold).item()
            dice_scores.append(dice)
            iou = iou_score(outputs, masks, threshold).item()
            iou_scores.append(iou)
            preds = (outputs > threshold).float().cpu().numpy().flatten()
            targets = masks.cpu().numpy().flatten()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_preds, average='binary', zero_division=0
    )
    accuracy = accuracy_score(all_targets, all_preds)

    metrics = {
        'dice': np.mean(dice_scores),
        'iou': np.mean(iou_scores),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }
    
    return metrics

def visualize_predictions(model, test_loader, device, threshold=0.5, num_samples=5, save_dir='results/layout'):
    """
    Visualize model predictions.
    
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= num_samples:
                break
                
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
                
            outputs = model(images)
                
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            images = images * std + mean
            
            images = images.cpu().numpy()
            masks = masks.cpu().numpy()
            outputs = outputs.cpu().numpy()
                
            #Create visualization
            for j in range(images.shape[0]):
                image = np.clip(images[j].transpose(1, 2, 0), 0, 1)
                image = (image * 255).astype(np.uint8)
                    
                mask = masks[j, 0]
                pred = (outputs[j, 0] > threshold).astype(np.float32)
                
                mask_overlay = image.copy()
                pred_overlay = image.copy()
                
                green_overlay = np.zeros_like(image)
                green_overlay[:, :, 1] = 255  #Full green channel
                
                red_overlay = np.zeros_like(image)
                red_overlay[:, :, 0] = 255  #Full red channel
                
                #Apply alpha blending ONLY where mask/prediction is positive
                alpha = 0.3  #Transparency level
                
                #For ground truth (green)
                mask_bool = mask > 0.5
                mask_overlay[mask_bool] = (1-alpha) * mask_overlay[mask_bool] + alpha * green_overlay[mask_bool]
                
                #For prediction (red)
                pred_bool = pred > 0.5
                pred_overlay[pred_bool] = (1-alpha) * pred_overlay[pred_bool] + alpha * red_overlay[pred_bool]

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(image)
                axes[0].set_title('Original Image')
                axes[0].axis('off')
                
                axes[1].imshow(mask_overlay)
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')
                
                axes[2].imshow(pred_overlay)
                axes[2].set_title('Prediction')
                axes[2].axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f'sample_{i}_{j}.png'))
                plt.close()

def evaluate_by_source(model, test_loader, device, threshold=0.5, save_dir='results/layout'):
    """
    Evaluate the model by source document.
    
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    
    metrics_by_source = {}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating by source"):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            sources = batch.get('source', ['unknown'] * images.shape[0])
            outputs = model(images)
            for i in range(images.shape[0]):
                image = images[i:i+1]
                mask = masks[i:i+1]
                output = outputs[i:i+1]
                source = sources[i]

                dice = dice_coefficient(output, mask, threshold).item()
                iou = iou_score(output, mask, threshold).item()
                if source not in metrics_by_source:
                    metrics_by_source[source] = {
                        'dice_scores': [],
                        'iou_scores': []
                    }
                
                metrics_by_source[source]['dice_scores'].append(dice)
                metrics_by_source[source]['iou_scores'].append(iou)

    results_by_source = {}
    for source, metrics in metrics_by_source.items():
        results_by_source[source] = {
            'dice': np.mean(metrics['dice_scores']),
            'iou': np.mean(metrics['iou_scores']),
            'num_samples': len(metrics['dice_scores'])
        }
    
    #Save results
    with open(os.path.join(save_dir, 'metrics_by_source.json'), 'w') as f:
        json.dump(results_by_source, f, indent=4)
    
    plot_metrics_by_source(results_by_source, save_dir)
    
    return results_by_source

def plot_metrics_by_source(metrics_by_source, save_dir):
    """
    Plot metrics by source.
    
    """
    sources = list(metrics_by_source.keys())
    dice_scores = [metrics_by_source[source]['dice'] for source in sources]
    iou_scores = [metrics_by_source[source]['iou'] for source in sources]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    #Plot Dice scores
    axes[0].bar(range(len(sources)), dice_scores)
    axes[0].set_title('Dice Coefficient by Source')
    axes[0].set_xlabel('Source')
    axes[0].set_ylabel('Dice Coefficient')
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(range(len(sources)))
    axes[0].set_xticklabels(sources, rotation=45, ha='right')

    #Plot IoU scores
    axes[1].bar(range(len(sources)), iou_scores)
    axes[1].set_title('IoU Score by Source')
    axes[1].set_xlabel('Source')
    axes[1].set_ylabel('IoU Score')
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(range(len(sources)))
    axes[1].set_xticklabels(sources, rotation=45, ha='right')


    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_by_source.png'))
    plt.close()

def main():
    torch.manual_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    batch_size = 16
    model_name = 'cnn_rnn_model'  # 'unet' or 'resnet_unet' or 'cnn_rnn_model'
    threshold = 0.5
    image_size = (512, 512)

    model_dir = os.path.join('models', 'layout')
    save_dir = os.path.join('results', 'layout')

    os.makedirs(save_dir, exist_ok=True)
    
    #Find latest model
    model_paths = [os.path.join(model_dir, d, 'best_model.pth') for d in os.listdir(model_dir) 
                   if os.path.isdir(os.path.join(model_dir, d))]
    model_paths = [p for p in model_paths if os.path.exists(p)]
    
    if not model_paths:
        print("No trained model found. Please train a model first.")
        return
    
    #Sort by modification time
    model_path = sorted(model_paths, key=lambda p: os.path.getmtime(p))[-1]
    print(f"Using model: {model_path}")

    model = load_model(model_path, model_name, device)

    prepared_data_dir = os.path.join('data', 'prepared')
    with open(os.path.join(prepared_data_dir, 'splits.json'), 'r') as f:
        splits = json.load(f)
    
    #Creating test dataset
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_data = []
    for image_path in splits['test']:
        mask_path = image_path.replace('_image.png', '_mask.png')
        parts = image_path.split(os.sep)
        source = parts[-2] if len(parts) > 2 else 'unknown'
        test_data.append({'image_path': image_path, 'mask_path': mask_path, 'source': source})
    
    test_dataset = LayoutDataset(test_data, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    #Evaluate model
    print("Evaluating model...")
    metrics = evaluate_model(model, test_loader, device, threshold)
    
    #Print metrics
    print("\nEvaluation Metrics:")
    print(f"Dice Coefficient: {metrics['dice']:.4f}")
    print(f"IoU Score: {metrics['iou']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    #Save
    with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    

    print("\nVisualizing predictions...")
    visualize_predictions(model, test_loader, device, threshold, num_samples=5, save_dir=save_dir)
    

    print("\nEvaluating by source...")
    metrics_by_source = evaluate_by_source(model, test_loader, device, threshold, save_dir)
    plot_metrics_by_source(metrics_by_source=metrics_by_source, save_dir=save_dir)
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
