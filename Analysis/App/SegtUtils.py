
import pickle
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
import cv2
from PIL import Image
import numpy as np
import torchvision.transforms as T
from Viz import UNet_Attention , load_checkpoint
import albumentations as A
from albumentations.pytorch import ToTensorV2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def Preprocessing(img):

    lab_image = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_l_channel = clahe.apply(l_channel)

    # Merge the CLAHE-enhanced L channel with the original A and B channels
    clahe_lab_image = cv2.merge((clahe_l_channel, a_channel, b_channel))

    # Convert the LAB image back to BGR color space
    clahe_bgr_image = cv2.cvtColor(clahe_lab_image, cv2.COLOR_LAB2BGR)

    # Convert the processed image back to PIL format
    image_pil_clahe = Image.fromarray(cv2.cvtColor(clahe_bgr_image, cv2.COLOR_BGR2RGB))
    return image_pil_clahe

def segmentation(img,frame, model, transform, device):
    img = Preprocessing(img)
    # Preprocess image
    img = np.array(img,dtype=np.float32)
    img = transform(image=img)["image"]
    img = img.unsqueeze(0).to(device)

    # Forward pass through the model
    model.eval()

    with torch.no_grad():
        out, _= model(img)

        # Post-process the output
        out = torch.sigmoid(out)
        out = (out > 0.5).float()

    out = out.squeeze(0)
    img = img.squeeze(0)

    frame = Preprocessing(frame)
    # Preprocess image
    frame = np.array(frame,dtype=np.float32)
    frame = transform(image=frame)["image"]

    transform = T.ToPILImage()

    mask = transform(out)
    gr = transform(img)
    image = transform(frame)

    mask_rgb = Image.merge('RGB', [mask, mask, mask])

    alpha = 0.6  # Set the desired transparency level
    overlay = Image.blend(image, mask_rgb, alpha)
    return  overlay