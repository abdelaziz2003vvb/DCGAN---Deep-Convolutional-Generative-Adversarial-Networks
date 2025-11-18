# DCGAN - Deep Convolutional Generative Adversarial Networks

PyTorch implementation of DCGAN based on the paper [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/abs/1511.06434) by Radford et al.

## Overview

This project implements a Deep Convolutional GAN (DCGAN) for generating images. The implementation follows the architectural guidelines from the original paper and includes training on the MNIST dataset.

## Architecture

### Generator
- Takes a 100-dimensional noise vector as input
- Uses transposed convolutions (stride=2) for upsampling
- Progressive feature map growth: 1x1 → 4x4 → 8x8 → 16x16 → 32x32 → 64x64
- ReLU activations in hidden layers
- Tanh activation in output layer
- Outputs 64x64 images

### Discriminator
- Takes 64x64 images as input
- Uses strided convolutions (stride=2) for downsampling instead of pooling
- Progressive feature map reduction: 64x64 → 32x32 → 16x16 → 8x8 → 4x4 → 1x1
- LeakyReLU(0.2) activations
- Sigmoid activation in output layer
- Outputs a single probability value (real vs fake)

## Key Features

Following DCGAN paper guidelines:
- ✅ No fully connected layers (except output)
- ✅ Strided convolutions for downsampling/upsampling
- ✅ Proper activation functions (LeakyReLU, ReLU, Tanh)
- ✅ Weight initialization (Normal distribution: mean=0.0, std=0.02)
- ⚠️ **Note**: Batch normalization is commented out in the current implementation

## Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
tensorboard
```

## Installation

```bash
pip install torch torchvision tensorboard
```

## Usage

### Training

```bash
python train.py
```

### Testing the Model Architecture

```bash
python model.py
```

This will run basic tests to verify the discriminator and generator architectures are working correctly.

### Monitoring Training

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir=logs
```

Then open your browser to `http://localhost:6006` to view:
- Real images from the dataset
- Generated (fake) images during training
- Training progression over time

## Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Learning Rate | 2e-4 | For both generator and discriminator |
| Batch Size | 128 | Number of images per batch |
| Image Size | 64x64 | Resolution of generated images |
| Noise Dimension | 100 | Input noise vector size |
| Epochs | 5 | Number of training epochs |
| Beta1 | 0.5 | Adam optimizer beta1 parameter |
| Beta2 | 0.999 | Adam optimizer beta2 parameter |

## Training Details

The training loop implements the standard GAN training procedure:

1. **Train Discriminator**: Maximize log(D(x)) + log(1 - D(G(z)))
   - Train on real images with label 1
   - Train on fake images with label 0

2. **Train Generator**: Maximize log(D(G(z)))
   - Generate fake images and train to fool discriminator

## Dataset

Default: MNIST dataset (1 channel, grayscale)
- Automatically downloads on first run
- Resized to 64x64
- Normalized to [-1, 1]

To use CelebA or other datasets:
```python
# Uncomment in train.py:
dataset = datasets.ImageFolder(root="celeb_dataset", transform=transforms)
# And change CHANNELS_IMG to 3 for RGB images
```

## Project Structure

```
.
├── model.py           # Generator and Discriminator architecture
├── train.py           # Training script
├── README.md          # This file
├── dataset/           # MNIST dataset (auto-downloaded)
└── logs/              # TensorBoard logs
    ├── real/          # Real images
    └── fake/          # Generated images
```

## Important Notes

### Batch Normalization
The current implementation has batch normalization **commented out** in both the generator and discriminator. According to the DCGAN paper, batch normalization is crucial for:
- Training stability
- Preventing mode collapse
- Better gradient flow

**Recommendation**: Uncomment the `nn.BatchNorm2d()` lines in both networks for better results:

```python
def _block(self, in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(...),
        nn.BatchNorm2d(out_channels),  # Uncomment this
        nn.LeakyReLU(0.2),
    )
```

### GPU Support
The code automatically uses CUDA if available, otherwise falls back to CPU.

## Results

After training, you should see:
- Discriminator loss stabilizing around 0.5-0.7
- Generator loss stabilizing around 0.7-1.5
- Generated images improving in quality over epochs

Check TensorBoard to visualize the progression of generated images.

## Tips for Better Results

1. **Enable Batch Normalization**: Uncomment BN layers for stable training
2. **Train Longer**: Increase `NUM_EPOCHS` for better quality (20-50 epochs)
3. **Label Smoothing**: Use 0.9 instead of 1.0 for real labels
4. **Learning Rate**: Experiment with different learning rates (1e-4 to 5e-4)
5. **Architecture**: Adjust `FEATURES_DISC` and `FEATURES_GEN` for capacity

## References

- [Original DCGAN Paper](https://arxiv.org/abs/1511.06434)
- Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.

## Author

Programmed by Aladdin Persson
- Email: aladdin.persson@hotmail.com
- Initial coding: 2020-11-01
- Revision: 2022-12-20

## License

This implementation is provided as-is for educational purposes.

## Troubleshooting

**Mode Collapse**: If generator produces same images repeatedly, try:
- Enabling batch normalization
- Reducing learning rate
- Adding label smoothing
- Training discriminator more frequently

**Training Instability**: If losses diverge, try:
- Enabling batch normalization
- Reducing learning rate
- Using gradient clipping
- Checking your data normalization

**Out of Memory**: Reduce `BATCH_SIZE` if you encounter CUDA OOM errors.
