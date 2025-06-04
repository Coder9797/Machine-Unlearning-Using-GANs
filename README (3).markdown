# Machine Unlearning Project

This project implements a machine unlearning framework using a Generative Adversarial Network (GAN) approach to remove specific data influences from a pre-trained ResNet-18 classifier. The goal is to mitigate the impact of certain training data (e.g., specific images) on the model's predictions while maintaining overall performance. Additionally, a membership inference attack is performed to evaluate the effectiveness of the unlearning process, achieving a False Negative Rate (FNR) score of 0.75.

## Project Overview

The project consists of three main Python scripts:
1. **Resnet_Model.py**: Prepares the dataset, defines and trains a modified ResNet-18 classifier, achieving a test accuracy of 84.81%, and saves the trained model weights and data batches.
2. **GANs.py**: Implements a GAN-based unlearning mechanism where a generator (ResNet-based) is trained to produce outputs indistinguishable from the original model, retaining a test accuracy of 63.35% after unlearning.
3. **Membership_Inference_Attack_Code.py**: Performs a membership inference attack to assess whether the unlearning process successfully removes the influence of specific data points, with an FNR score of 0.75.

The project uses PyTorch and torchvision. The dataset is assumed to be CIFAR-10 or a similar image classification dataset with 10 classes.

## Prerequisites

To run this project, ensure you have the following installed:

- Python 3.8+
- PyTorch
- torchvision
- pandas
- PIL (Pillow)
- scikit-learn
- numpy
- tqdm
- (Optional) wandb for logging metrics
- (Optional) py7zr for extracting compressed dataset files

You can install the required packages using:
```bash
pip install torch torchvision pandas pillow scikit-learn numpy tqdm
pip install wandb py7zr
```

## Project Structure

- **Resnet_Model.py**: 
  - Loads and preprocesses the dataset (e.g., CIFAR-10) using transformations like random cropping and normalization.
  - Defines a modified ResNet-18 model with a custom first convolutional layer and no max pooling.
  - Trains the model for 10 epochs, achieving a test accuracy of 84.81%, and saves weights as `resnet18_weights.pth`.
  - Saves data batches for training and testing as `.pt` files (e.g., `first_batch.pt`, `remaining_batches.pt`, `train_loader_batches.pt`, `test_loader_batches.pt`, `train_batch.pt`, `test_batch.pt`).

- **GANs.py**:
  - Defines a `ResNetClassifierGenerator` (generator) and `ProbabilityDiscriminator` (discriminator) for the GAN-based unlearning process.
  - Loads pre-trained ResNet-18 weights and trains the generator to mimic the original model's output distribution while unlearning specific data influences, retaining a test accuracy of 63.35%.
  - Uses Wasserstein GAN with gradient penalty for stable training.
  - Saves the unlearned model weights as `model_b_weights7.pth`.

- **Membership_Inference_Attack_Code.py**:
  - Implements a membership inference attack to evaluate whether specific data points (e.g., `first_batch.pt`) have been successfully unlearned, achieving an FNR score of 0.75.
  - Trains an attack model to distinguish between member (training) and non-member (test) data based on the model's output probabilities.
  - Reports metrics like accuracy, precision, recall, F1 score, and FNR to assess unlearning effectiveness.

## Dataset

The project assumes a dataset structured like CIFAR-10, with images stored in a directory (`Extracted_Images/train`) and labels in a CSV file (`trainLabels.csv`) with columns `id` and `label`. The dataset is split into training (80%) and validation (20%) sets. Data batches are saved as `.pt` files for use in the unlearning and attack phases.

To prepare the dataset:
1. If using a compressed dataset (e.g., `train.7z`), extract it using the commented code in `Resnet_Model.py`.
2. Ensure `trainLabels.csv` contains image IDs and corresponding labels.
3. The dataset is loaded and transformed using `torchvision.transforms`.

## Usage

1. **Train the Initial Model**:
   Run `Resnet_Model.py` to train the ResNet-18 classifier and save the weights and data batches:
   ```bash
   python Resnet_Model.py
   ```
   - Outputs: `resnet18_weights.pth`, `first_batch.pt`, `remaining_batches.pt`, `train_loader_batches.pt`, `test_loader_batches.pt`, `train_batch.pt`, `test_batch.pt`.
   - Achieves a test accuracy of 84.81%.
   - Optional: Configure `wandb` for logging by providing your API key.

2. **Perform Machine Unlearning**:
   Run `GANs.py` to train the GAN-based unlearning model, which removes the influence of specific data (e.g., `first_batch.pt`):
   ```bash
   python GANs.py
   ```
   - Outputs: `model_b_weights7.pth` (unlearned model weights with 63.35% test accuracy), `real_prob_test.pt` (sorted probabilities from the original model).
   - The script trains for 10 epochs with a Wasserstein GAN and gradient penalty.

3. **Evaluate Unlearning with Membership Inference Attack**:
   Run `Membership_Inference_Attack_Code.py` to perform a membership inference attack and assess unlearning effectiveness:
   ```bash
   python Membership_Inference_Attack_Code.py
   ```
   - Outputs: Metrics (accuracy, precision, recall, F1, FNR=0.75) for the attack on training, test, and deleted (unlearned) data.
   - The script evaluates whether the unlearned model can distinguish member (training) from non-member (test) data.

## Results

- **ResNet-18 Training**: The initial ResNet-18 model is trained for 10 epochs, achieving a test accuracy of 84.81%, with training and validation accuracy/loss logged (via `wandb` if enabled).
- **Unlearning**: The GAN-based unlearning process trains a generator to produce outputs indistinguishable from the original model while minimizing the influence of specific data, retaining a test accuracy of 63.35%.
- **Membership Inference Attack**:
  - Achieves an FNR score of 0.75, indicating effective unlearning.
  - For training data (`x_trains`): High membership prediction (label=1) indicates the model retains training data information.
  - For test data (`x_test`, `x_test1`): High non-membership prediction (label=0) with FNR scores of 0.78 and 0.75 indicates successful unlearning.
  - For deleted data (`x_test_delete`): High non-membership prediction (label=0) suggests the unlearning process effectively removed the influence of these data points.

## Notes

- Ensure the file paths for saved weights and data batches match your local setup.
- The commented `wandb` code requires a valid API key for logging. Remove or configure as needed.
- The membership inference attack assumes balanced datasets for training and testing to ensure fair evaluation.
- The project uses a modified ResNet-18 architecture tailored for CIFAR-10-like datasets (32x32 images, 10 classes).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built using PyTorch and torchvision libraries.
- Inspired by research on machine unlearning and membership inference attacks.
- Dataset handling adapted for CIFAR-10 or similar image classification tasks.