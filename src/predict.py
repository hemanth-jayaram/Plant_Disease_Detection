import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import json

def get_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

def load_model(model_path, num_classes=3):
    # Load ResNet50 with the same architecture as during training
    model = resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    
    # Load the saved model weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(image_path, model_path, class_names=None):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(model_path, num_classes=3 if class_names is None else len(class_names))
    model = model.to(device)
    
    # Prepare image
    # Prepare image
    transform = get_transform()
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Get class name if available
    if class_names is not None:
        predicted_class = class_names[predicted.item()]
    else:
        predicted_class = f'Class {predicted.item()}'
    
    # Get confidence score
    confidence = probabilities[predicted].item() * 100
    
    # Get top 3 predictions if class names are provided
    top_predictions = []
    if class_names is not None:
        top3_prob, top3_catid = torch.topk(probabilities, 3)
        for i in range(top3_prob.size(0)):
            top_predictions.append({
                'class': class_names[top3_catid[i].item()],
                'probability': top3_prob[i].item() * 100
            })
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'class_id': predicted.item(),
        'top_predictions': top_predictions if class_names else None
    }

def display_prediction(image_path, prediction_result, class_names):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    
    # Create figure and axis
    plt.figure(figsize=(10, 8))
    
    # Display the image
    plt.imshow(image)
    plt.axis('off')
    
    # Prepare text to display
    pred_text = f"Predicted: {prediction_result['predicted_class']} ({prediction_result['confidence']:.1f}%)"
    
    # Add prediction text below the image
    plt.figtext(0.5, 0.01, pred_text, 
               ha='center', fontsize=14, 
               bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Add top predictions as a table
    if prediction_result['top_predictions']:
        table_data = [[f"{i+1}. {p['class']}", f"{p['probability']:.1f}%"] 
                     for i, p in enumerate(prediction_result['top_predictions'])]
        
        plt.table(cellText=table_data,
                 colLabels=['Class', 'Probability'],
                 cellLoc='center',
                 loc='bottom',
                 bbox=[0.1, -0.3, 0.8, 0.2])
    
    plt.tight_layout()
    plt.show()

def main():
    import argparse
    import os

    # Define class names (should match the order from training)
    CLASS_NAMES = ['Healthy', 'Powdery', 'Rust']

    # Use a relative default image path (place a sample image in the repo)
    DEFAULT_IMAGE_PATH = os.path.join('data', 'val', 'Rust', '963ffc6b98d60940.jpg')

    # Use a relative model path
    DEFAULT_MODEL_PATH = os.path.join('models', 'best_model.pth')

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Predict plant disease from an image')
    parser.add_argument('--image_path', type=str, default=DEFAULT_IMAGE_PATH,
                        help='Path to the image file')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to the trained model')
    parser.add_argument('--no-display', action='store_true',
                        help='Disable displaying the image with prediction')

    args = parser.parse_args()

    # Make prediction
    img_path = args.image_path
    print(f"\nUsing image: {img_path}")
    result = predict_image(img_path, args.model, CLASS_NAMES)

    # Print results
    print(f"\nPrediction Results:")
    print(f"- Predicted class: {result['predicted_class']}")
    print(f"- Confidence: {result['confidence']:.2f}%")

    if result['top_predictions']:
        print("\nTop 3 predictions:")
        for i, pred in enumerate(result['top_predictions']):
            print(f"{i+1}. {pred['class']}: {pred['probability']:.2f}%")

    # Display the image with prediction if not disabled
    if not args.no_display:
        display_prediction(args.image_path, result, CLASS_NAMES)

if __name__ == '__main__':
    main()

