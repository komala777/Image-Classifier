# predict.py for AWS AI and ML - Udacity Scholarship program
import argparse
import torch
from torchvision import transforms, models
from torch import nn
from PIL import Image
import json
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    model = models.vgg16(pretrained=True)
    
    if 'classifier' in checkpoint:
        model.classifier = checkpoint['classifier']
    else:
       
        classifier = nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 102),
            nn.LogSoftmax(dim=1)
        )
        model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model
def process_image(image_path):
  
    img = Image.open(image_path)
 
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
  
    img_tensor = transform(img)
    return img_tensor
def predict(image_tensor, model, topk=3, category_names=None, gpu=False):

    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    model.eval()
    
    with torch.no_grad():
        output = model(image_tensor)
    
    probs, classes = torch.exp(output).topk(topk)
  
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx] for idx in classes.cpu().numpy()[0]]
   
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_classes = [cat_to_name[class_label] for class_label in top_classes]
    return probs.cpu().numpy()[0], top_classes
def main():
    parser = argparse.ArgumentParser(description='Predict flower ')
    parser.add_argument('image_path', type=str, help='Path ')
    parser.add_argument('checkpoint', type=str, help='Path - checkpoint')
    parser.add_argument('--top_k', type=int, default=3, help='Top K ')
    parser.add_argument('--category_names', type=str, default=None, help='Path - JSON file ')
    parser.add_argument('--gpu', action='store_true', help='GPU ')
    args = parser.parse_args()
    
    model = load_checkpoint(args.checkpoint)
   
    image_tensor = process_image(args.image_path)
    
    top_probs, top_classes = predict(image_tensor, model, topk=args.top_k, category_names=args.category_names, gpu=args.gpu)
    
    print("Classes are the :", top_classes)
    print("Probability values:", top_probs)
if __name__ == '__main__':
    main()                         




