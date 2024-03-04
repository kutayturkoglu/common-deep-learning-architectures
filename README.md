# Deep Learning Image Classification from Scratch

This repository contains an implementation of common deep learning methods for image classification from scratch using PyTorch, NumPy, and Pandas. The implemented methods include a custom-built data loader, training loop, and layers, allowing for a comprehensive understanding and customization of the entire process.

## Dataset
The dataset used for this project is sourced from Kaggle, specifically from the Intel Image Classification dataset available at [this link](https://www.kaggle.com/datasets/puneet6060/intel-image-classification). It consists of 25,000 images categorized into 6 classes, with each image having dimensions of 150x150 pixels. Since the target architectures require specific input dimensions, preprocessing steps are applied to modify the dataset accordingly.

## Implemented Models
Currently, the repository contains the implementation of VGG16 architecture. However, the plan is to expand the collection by adding other popular architectures such as VGG19, ResNet, AlexNet, Google LeNet, etc. This expansion aims to provide a diverse range of architectures for experimentation and comparison purposes.

## Requirements
- PyTorch
- NumPy
- tqdm
- sys
- os

## Usage
1. **Dataset Preparation**: Download the dataset from the provided Kaggle link and ensure it is placed in the appropriate directory. Preprocess the images to fit the desired input dimensions.
   
2. **Training**: Execute the training script by specifying the desired parameters such as batch size, learning rate, number of epochs, etc. This script will train the chosen model architecture on the provided dataset.

3. **Evaluation**: After training, evaluation scripts can be run to assess the performance of the trained model on validation or test datasets. Metrics such as accuracy, precision, recall, and F1 score can be calculated to gauge the model's effectiveness.

4. **Model Expansion**: To add additional model architectures, implement the desired architecture following the structure of the existing models. Ensure proper integration with the training loop and data loader for seamless experimentation.

## Contributions
Contributions to this repository are welcome! If you have implemented additional model architectures, optimized training procedures, or improved the overall codebase, feel free to submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments
- Intel for providing the dataset used in this project.
- PyTorch, NumPy, and Pandas communities for their invaluable contributions to the field of deep learning and data manipulation.
