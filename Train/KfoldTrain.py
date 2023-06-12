import pickle
import torch
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import SubsetRandomSampler
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from Models.AttUnet import UNet_Attention
from sklearn.model_selection import KFold
# from Loss import DiceLoss
# from monai.losses import DiceLoss
from pytorch_toolbelt.losses import DiceLoss
from torch.utils.tensorboard import SummaryWriter
from Utils.KfoldUtils import (
    load_checkpoint,
    save_checkpoint,
    get_Data,
    check_accuracy,
    save_predictions_as_imgs,
    save_map
)

LEARNING_RATE = 1e-4
DECAY = 1e-5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 8
KFOLDS = 10
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 512
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "../data/train/Images"
TRAIN_MASK_DIR = "../data/train/teeth_mask"
VAL_IMG_DIR = "../data/val/Images"
VAL_MASK_DIR = "../data/val/teeth_mask"


# TODO: visualize layers https://ravivaishnav20.medium.com/visualizing-feature-maps-using-pytorch-12a48cd1e573

# Configuration options

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = LEARNING_RATE * (0.8 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions,_ = model(data)
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
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
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
            # A.Rotate(limit=35, p=1.0),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )
    train_ds, test_ds = get_Data(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        train_transform,
        val_transforms,
    )

    dataset = ConcatDataset([train_ds, test_ds])
    torch.manual_seed(42)
    kfold = KFold(n_splits=KFOLDS, shuffle=True)
    resultsPA = {}
    resultsIoU = {}
    resultsDice = {}
    model = UNet_Attention().to(DEVICE)
    loss_fn = DiceLoss(mode='binary', log_loss=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scaler = torch.cuda.amp.GradScaler()
    global_step = 0

    if LOAD_MODEL:
        load_checkpoint(torch.load("../my_checkpoint.pth.tar"), model)

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):

        print('--------------------------------')
        print(f'FOLD {fold + 1}')
        print('--------------------------------')

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        test_subsampler = SubsetRandomSampler(test_ids)

        # Define data loaders for training and testing data in this fold
        trainloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            sampler=train_subsampler
        )

        testloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            num_workers=PIN_MEMORY,
            pin_memory=PIN_MEMORY,
            sampler=test_subsampler
        )

        # Run the training loop for defined number of epochs
        for epoch in range(0, NUM_EPOCHS):
            print(f'Starting epoch {epoch + 1}')
            train_fn(trainloader, model, optimizer, loss_fn, scaler)
            adjust_learning_rate(optimizer, epoch)

            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }

            save_checkpoint(checkpoint)
            # Print about testing
            print('Starting testing')
            pa, dice, iou, = check_accuracy(testloader, fold, model, device="cuda")
            resultsPA[fold] = pa
            resultsDice[fold] = dice
            resultsIoU[fold] = iou
            writer.add_scalar('Val Dice Score', dice, global_step=global_step)
            writer.add_scalar('Val Accuracy', pa, global_step=global_step)
            writer.add_scalar('Val IoU', iou, global_step=global_step)
            global_step += 1
            save_predictions_as_imgs(
                testloader, model, folder="../saved_images/", device=DEVICE
            )
            save_map('../data/val/Images/5.JPG', model, val_transforms, DEVICE, global_step)

    print(f'TRAINING FINISHED FOR ALL {KFOLDS} FOLDS!!!!!!!!')
    print('====================================================================================')
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {KFOLDS} FOLDS')
    print('--------------------------------')
    sum1 = 0.0
    for key1, value1 in resultsPA.items():
        print(f'Fold {key1}: {value1:.2f} %')
        sum1 += value1
    print(f'Average PA: {sum1 / len(resultsPA.items()):.2f} %')
    print('====================================================================================')
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {KFOLDS} FOLDS')
    print('--------------------------------')
    sum2 = 0.0
    for key2, value2 in resultsIoU.items():
        print(f'Fold {key2}: {value2:.2f} %')
        sum2 += value2
    print(f'Average: {sum2 / len(resultsIoU.items()):.2f} %')

    print('====================================================================================')
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {KFOLDS} FOLDS')
    print('--------------------------------')
    sum3 = 0.0
    for key3, value3 in resultsDice.items():
        print(f'Fold {key3}: {value3:.2f} %')
        sum3 += value3
    print(f'Average: {sum3 / len(resultsDice.items()):.2f} %')
    print('====================================================================================')

    with open('../Analysis/PA.pkl', 'wb') as file:
        print('Saving Final Results PA...')
        pickle.dump(resultsPA, file)

    file.close()
    with open('../Analysis/IoU.pkl', 'wb') as file:
        print('Saving Final Results IoU...')
        pickle.dump(resultsIoU, file)
    file.close()

    with open('../Analysis/Dice.pkl', 'wb') as file:
        print('Saving Final Results Dice...')
        pickle.dump(resultsDice, file)

    file.close()
    print('Done!')


if __name__ == "__main__":
    main()
