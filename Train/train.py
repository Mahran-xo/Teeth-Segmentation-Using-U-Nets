import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.optim as optim
from Models.AttUnet import UNet_Attention
from pytorch_toolbelt.losses import DiceLoss
from torch.utils.tensorboard import SummaryWriter


from Utils.utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs
)

# hyperparams
LEARNING_RATE = 1e-4
DECAY = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
NUM_EPOCHS = 200
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "../data/train/Images"
TRAIN_MASK_DIR = "../data/train/teeth_mask"
VAL_IMG_DIR = "../data/val/Images"
VAL_MASK_DIR = "../data/val/teeth_mask"


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = LEARNING_RATE * (0.8 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])


def main():
    writer = SummaryWriter(f'../runs/Data/tensorborad')

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNet_Attention().to(DEVICE)
    loss_fn = DiceLoss(mode='binary',log_loss=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("../my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)
        adjust_learning_rate(optimizer, epoch)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        save_checkpoint(checkpoint)

        acc, dice, iou = check_accuracy(val_loader, model, device=DEVICE)
        writer.add_scalar('Val Dice Score', dice, global_step=global_step)
        writer.add_scalar('Val Accuracy', acc, global_step=global_step)
        writer.add_scalar('Val IoU', iou, global_step=global_step)

        global_step += 1

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="../saved_images/", device=DEVICE
        )


if __name__ == "__main__":
    main()
