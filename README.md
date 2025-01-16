# Generating conformational ensemble using a Denoising Diffusion Probabilistic Model (DDPM)

Welcome to the repository for the research project titled **Assessing Generative Diffusion Models for Enhanced Sampling of Folded and
Disordered Protein States Across Scales and in All-atom Resolution**

### The code has been adopted from the following GitHub repositories and websites, with necessary modifications.

- [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)  
- [tiwarylab/DDPM_REMD](https://github.com/tiwarylab/DDPM_REMD)  
- [Hugging Face Blog - Annotated Diffusion](https://huggingface.co/blog/annotated-diffusion)  

We sincerely thank the authors of these resources for their valuable contributions.

# Code Requirements

To run the code, ensure you have the following Python packages installed:

- [numpy](https://numpy.org/)
- [torch](https://pytorch.org/)

# A schematic overview of the noising and denoising processes.

We illustrated the noising-denoising processes
using an image of a Paperflower clicked at the TIFRH campus during summer. The upper panel depicts the forward (noising)
process, represented by the blue arrow, where the original input image $x_0$ is progressively corrupted by adding random noise,
and the reverse (denoising) process, represented by the black arrow, where a neural network predicts the added noise. The
lower panel illustrates the U-Net model architecture.

![](schematic_git.png)

## Directory Structure & Usage

### 1. **Training the Model**
   - To train the model, run the script:
     ```bash
     python model_train.py
     ```
   - After the model is trained, a folder named `results` will be created. Inside this folder, you will find subfolders containing the trained model.

### 2. **Generating Samples**
   - Once the model is trained, use the following script to generate samples:
     ```bash
     python sample_generate.py
     ```
   - The generated samples will be saved in a folder named `generate_sample`.

### 3. **Provided Example**
   - We have provided a trained model for the "moon" dataset. The folder `moon` contains training data in `.npy` format with two axes: x and y coordinates.
   - We have also included the backbone torsion data for Trp-cage mini-protein. The folder `Trpcage` contains the training data in `.npy` format.

### 4. **Data Types Supported**
   - This code is versatile and can be applied to various types of data, here are some examples:
     - Torsion angles
     - Raw coordinates of all atoms
     - Protein-ligand distances
   - **Note**: For noise prediction, the code utilizes a 1D-UNET model. Ensure that the data for each frame is represented as a 1D array.

### 5. **Core Neural Network Code**
   - The folder `denoising_diffusion_1D` contains the main code for the 1D neural network architecture used for noising and denoising the data.

## Animation

- Here is a nice animation demonstrating how moon data can be generated from pure random noise using DDPM.

![Animation](animation.gif)

## News!
