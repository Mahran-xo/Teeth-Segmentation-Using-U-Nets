# Teeth Segmentation using U-Net Variants on TUFTS Dataset

This repository contains code for teeth segmentation using variants of U-Net on the TUFTS dataset using PyTorch. Please note that this repository is still under construction.

## Dataset

The TUFTS dataset contains X-ray images of teeth. Each image has corresponding ground truth masks for teeth segmentation.

## U-Net Variants

I'am are experimenting with different variants of U-Net for teeth segmentation. These include:

- Vanilla U-Net
- Attention U-Net

## Usage

To train the models, please follow these steps:

1. Clone the repository
2. Install the required packages listed in `requirements.txt`
3. Download the TUFTS dataset and place it in the `data` directory
4. Run the training script `train.py`

## Results

Here are some example results from my experiments:

### Vanilla U-Net

True Label
![3](https://user-images.githubusercontent.com/96589883/233794612-661250f8-8134-4cdb-a801-b95be963296b.png)

Predicted
![pred_3](https://user-images.githubusercontent.com/96589883/233794620-01d93b38-4e1d-478e-9e4c-bb494843fec3.png)



### Attention U-Net

True Label
![13](https://user-images.githubusercontent.com/96589883/233794567-53ef5b3f-7541-4db8-88e3-b42dbafa075f.png)

Predicted
![pred_13](https://user-images.githubusercontent.com/96589883/233794577-afa22117-6013-491a-a1e6-57d55a47f0d3.png)




## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
