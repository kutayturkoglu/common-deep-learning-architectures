# ResNet Architecture README
![ResNet Architecture](https://miro.medium.com/v2/resize:fit:606/1*o7IEqUjg318onKRAhqO0Tg.png)
link: https://miro.medium.com/v2/resize:fit:606/1*o7IEqUjg318onKRAhqO0Tg.png
## Architecture Overview

This repository contains an implementation of a ResNet (Residual Network). The implementation focuses on simplicity and ease of use, incorporating only the essential components necessary for understanding the ResNet architecture.

## Residual Block Structure

The core building block of the ResNet architecture is the residual block. Each residual block consists of two convolutional layers, with optional downsampling to match dimensions. Here's a breakdown of the components within each residual block:

- **Convolutional Layers**: Two 2D convolutional layers are employed within each residual block. These layers are responsible for learning feature representations.
- **Batch Normalization**: Batch normalization is applied after each convolutional layer to stabilize and accelerate the training process.
- **ReLU Activation**: ReLU activation functions are used after each convolutional layer to introduce non-linearity into the network.
- **Downsampling**: Optionally, a downsampling operation may be applied if the input and output dimensions need to be matched. This is typically achieved using a 1x1 convolutional layer followed by batch normalization.

## ResNet Architecture

The ResNet architecture consists of several components:

- **Initial Convolutional Layer**: The network begins with a convolutional layer that processes the input data.
- **Max Pooling**: Max pooling is applied after the initial convolutional layer to downsample the feature maps.
- **Residual Layers**: The residual layers are composed of multiple residual blocks. The number of residual blocks can be adjusted based on the desired complexity of the network.
- **Adaptive Average Pooling**: An adaptive average pooling layer is employed to ensure that the output has a fixed size, regardless of the input size.
- **Fully Connected Layer**: Finally, a fully connected layer is used for classification, mapping the extracted features to the desired number of output classes.


