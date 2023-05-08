import argparse
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import json

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['architecture'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image)
    image = image_transforms(image)
    return image

def predict(image_path, model, topk, cat_to_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    image = process_image(image_path)
    image = image.unsqueeze(0)
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        probabilities = torch.exp(output)
        top_probabilities, top_indices = torch.topk(probabilities, topk)
        top_probabilities = top_probabilities.squeeze().tolist()
        top_indices = top_indices.squeeze().tolist()
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_labels = [cat_to_name[idx_to_class[idx]] for idx in top_indices]
    return top_probabilities, top_labels

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image.")
    parser.add_argument("image_path", type=str, help="Path to the image.")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint file.")
    parser.add_argument("--top_k", type=int, default=5, help="Return top K most likely classes.")
    parser.add_argument("--category_names", type=str, default="cat_to_name.json", help="Path to the mapping of categories to real names.")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference if available.")
    args = parser.parse_args()

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    model = load_checkpoint(args.checkpoint)
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = model.to(device)

    top_probabilities, top_labels = predict(args.image_path, model, args.top_k, cat_to_name)

    print(f"Top {args.top_k} predictions:")
    for i in range(args.top_k):
        print(f"  {top_labels[i]}: {top_probabilities[i]:.3f}")

if __name__ == '__main__':
    main()
