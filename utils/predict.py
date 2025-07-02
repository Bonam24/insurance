import torch
import timm
from PIL import Image
from torchvision import transforms

def load_all_models():
    """Load all pre-trained models for ensemble prediction"""
    models = {
        "moire": {
            "path": "models/new_moire_detection_model_with_metadata.pt",
            "threshold": 0.00077  # Special low threshold only for moire model
        },
        "computer": {
            "path": "models/computer_screen_detection_model_with_metadata.pt",
            "threshold": 0.5  # Default threshold
        },
        "phone": {
            "path": "models/phone_screen_detection_model_with_metadata.pt",
            "threshold": 0.5
        },
        "printed": {
            "path": "models/printed_sources_detection_model_with_metadata.pt",
            "threshold": 0.5
        },
        "tv": {
            "path": "models/tv_screen_detection_model_with_metadata.pt",
            "threshold": 0.5
        }
    }
    
    model_info = {}
    
    for name, config in models.items():
        try:
            checkpoint = torch.load(config["path"], map_location=torch.device('cpu'))
            
            # Initialize model (assuming all are ViT Tiny)
            model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Get class names and ensure correct order
            class_names = checkpoint.get('class_names', ['recaptured', 'real'])
            
            # Create transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            
            model_info[name] = {
                "model": model,
                "transform": transform,
                "class_names": class_names,
                "threshold": config["threshold"]  # Use the specific threshold for each model
            }
        except Exception as e:
            print(f"Error loading {name} model: {e}")
            continue
    
    return model_info

def predict_image(image_path, model_info):
    """Predict using ensemble of models with model-specific thresholds"""
    img = Image.open(image_path).convert('RGB')
    predictions = {}
    final_verdict = "camera"  # Default to camera unless any model says recaptured
    
    for name, info in model_info.items():
        try:
            img_tensor = info["transform"](img).unsqueeze(0)
            
            with torch.no_grad():
                output = info["model"](img_tensor)
                probabilities = torch.softmax(output, dim=1).squeeze()
                
                # Get indices based on class order
                moire_idx = info["class_names"].index('recaptured')
                real_idx = info["class_names"].index('real')
                
                moire_confidence = probabilities[moire_idx].item()
                real_confidence = probabilities[real_idx].item()
                
                # Apply model-specific threshold
                if moire_confidence >= info["threshold"]:
                    model_verdict = "recaptured"
                    final_verdict = "screen"  # Any recaptured changes final verdict
                else:
                    model_verdict = "real"
                
                predictions[name] = {
                    "class": model_verdict,
                    "moire_confidence": f"{moire_confidence:.6f}",
                    "real_confidence": f"{real_confidence:.6f}",
                    "threshold": info["threshold"],
                    "model_name": name.replace("_", " ").title()
                }
                
        except Exception as e:
            print(f"Error predicting with {name} model: {e}")
            predictions[name] = {
                "error": str(e),
                "model_name": name.replace("_", " ").title()
            }
    
    return {
        "model_predictions": predictions,
        "final_prediction": final_verdict,
        # Add the specific moire threshold for display purposes
        "moire_threshold": model_info["moire"]["threshold"] if "moire" in model_info else 0.00077
    }