import torch
import torchvision
from Dataset.dataset import TuftsDataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import cv2

from segmentation_models_pytorch.utils.metrics import IoU


def save_checkpoint(state, filename="../my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_Data(
        train_dir,
        train_maskdir,
        val_dir,
        val_maskdir,
        train_transform,
        val_transform,
):
    train_ds = TuftsDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    val_ds = TuftsDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    return train_ds, val_ds


def check_accuracy(testloader, fold, model, device="cuda"):
    jaccard = IoU(threshold=0.5)
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    iou = 0
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            x, y = data
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            prediction, _ = model(x)
            preds = torch.sigmoid(prediction)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            iou += jaccard(preds, y)
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )

    print(f"Dice score: {dice_score / len(testloader) * 100:.2f} for fold {fold}")
    print(f"IoU: {iou / len(testloader) * 100:.2f} for fold {fold}")
    print(f"PA: {num_correct / num_pixels * 100:.2f} for fold {fold}")
    model.train()
    return num_correct / num_pixels * 100, dice_score / len(testloader) * 100, iou / len(testloader) * 100,


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            prediction, _ = model(x)
            preds = torch.sigmoid(prediction)
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()


def get_map(att):
    feature_map = att.squeeze(0)

    gray_scale = torch.mean(feature_map, 0)

    gray_scale = gray_scale.data.cpu().numpy()

    gray_scale = (gray_scale - np.min(gray_scale)) / (np.max(gray_scale) - np.min(gray_scale))

    return gray_scale


def save_map(path, model, transform, DEVICE, save):
    image = np.array(Image.open(path).convert("RGB"))
    image = transform(image=image)["image"]
    model.eval()
    with torch.no_grad():
        x = image.float().unsqueeze(0).to(DEVICE)
        _, x3 = model(x)
    heatmap = get_map(x3)
    img = cv2.imread(path)
    img = cv2.resize(img, (512, 256))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img, 0.5, heatmap, 0.5, 0)
    cv2.imwrite(f'../saved_images/Attention Maps/{save}.jpg', heatmap)
    model.train()
