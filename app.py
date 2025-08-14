from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.transforms as transforms
from transformers import SwinForImageClassification
import os

from PIL import Image

from flask_cors import CORS #react frontend and flask backend are running on a diff port

app=Flask(__name__)
CORS(app, origins=["https://medi-scan-ai-frontend.vercel.app"])

#load the pretrained swin model for both kidney and brain
def load_model(model_path, num_of_classes):
    model = SwinForImageClassification.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
    model.classifier=nn.Linear(model.classifier.in_features, num_of_classes)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu"))) #load the model weights
    model.eval()  # model set to evaluation mode
    return model

#load both models
kidney_model=load_model('./swin_kidney_ct_scan_detection.pth',4)
brain_model=load_model('./swin_brain_tumor_detection.pth', 4)

#image tranformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize all images to 224x224 (Swin Transformer input size)
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@app.route('/predict',methods=['POST'])
def predict():
    file=request.files["file"]
    model_type=request.form["model_type"].lower()

    #select the appropriate model
    if model_type=="kidney":
        model=kidney_model
        class_labels=["Cyst", "Normal", "Stone", "Tumor"]
    elif model_type=="brain":
        model=brain_model
        class_labels=["Glioma", "Healthy", "Meningioma", "Pituitary"]

    image=Image.open(file).convert("RGB")
    image=transform(image).unsqueeze(0) #model expects batches of images and not just 1 image, therefore it adds a new dim
    #perform inference
    with torch.no_grad():
        output=model(image)

    probabilities=F.softmax(output.logits, dim=1) #converting raw prediction scores to probablities
    predicted_class=torch.argmax(probabilities, dim=1).item() #finds the index of the highest probablity
    confidence=probabilities[0, predicted_class].item()

    return jsonify({
        "prediction": class_labels[predicted_class],
        "confidence": confidence
    })

if __name__=="__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)