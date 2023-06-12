from monai.networks.nets import AttentionUnet
from segmentation_models_pytorch import Unet
# from pytorch_toolbelt.losses import DiceLoss
from monai.losses import DiceLoss

import pickle


with open('../Metrics.pkl', 'rb') as file:
    my_dict = pickle.load(file)



for key, value in my_dict.items():
    print(f'{key}: {value}')




# Unet(encoder_name="resnet50",
#      decoder_attention_type="scse"
#
#      )
#
# AttentionUnet(spatial_dims=2,
#               in_channels=3,
#               out_channels=1,
#               channels=(32, 64, 128, 256, 512),
#               strides=(2, 2, 2, 2),
#               dropout=0.5)

# train_transform = A.Compose(
#     [
#         A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
#         A.Rotate(limit=35, p=1.0),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.1),
#         A.Normalize(
#             mean=[0.0],
#             std=[1.0],
#             max_pixel_value=255.0,
#         ),
#         ToTensorV2(),
#     ],
# )
