# Generative Art Project
Final project for the 2024 Postgraduate course on
Artificial Intelligence with Deep Learning, 
UPC School, authored by 
**María Pía Sánchez**, 
**David Arméstar**,
**Bruno Pardo** and 
**Eduardo Torres**. 

Advised by **Laia Tarrès**.

Table of Contents
=================
  * [Introduction and motivation](#introduction-and-motivation)
  * [Dataset](#dataset)
    * [Description](#description) 
    * [How to prepare the dataset from scratch](#how-to-prepare-the-dataset-from-scratch)
      * [How to transform videos to poses](#how-to-extract-features-from-videos)
      * [Clean dataset](#clean-dataset)
      * [Convert video frames to video format](#convert-video-frames-to-video-format)
  * [Music to poses](#music-to-poses)
	 * [GANs components](#gan_components)
	 	* [Generator](#Generator)
	 	* [Discriminator](#rdiscriminator)
	 	* [Losses](#losses)
     * [Hyperparameter Tuning](#hyperparameter-tuning)
		* [Lerning Rate](#learning-rate)
        * [Number of Epochs](#number-of-epochs)
        * [Increasing Discriminator Layers](#increasing-discriminator-layers)
        * [Architectural Changes](#architectural-changes)
        * [Batch Size](#batch-size)
        * [Pixel Weighted Loss](#pixel-weighted-loss)
     * [Final results and conclusions](#final-results-conclusions)
  * [Poses to video](#poses-to-video)
    * [SD components](#sd-components)
       * [AnimateDiff SD model](#animatediff)
       * [Poses ControlNet](#poses-controlnet)
       * [TemporalNet](#temporalnet)
    * [Final results and conlusions](#final-results-conclusions)
  * [End to end pipeline](#end-to-end-system)
	 * [Platform architecture](#platform-architecture)
  * [How to](#how-to)
     * [Install the project](#installation)
        * [Install requirements](#install-requirements)
        * [Install conda](#install-miniconda)
        * [Create your conda environment](#create-your-miniconda-environment)
        * [Setting the environment in Google Drive](#setting-the-environment-in-google-drive)
	 * [RUN the project](#run-the-project)
        * [Running training-inferece scripts locally with conda](#run-wtih-conda)
        * [Running training-inferece scripts from VM in GCP](#run-in-gcp)
---
---

## Introduction and motivation

The main goal of this project is to deliver artistic visual results from raw audio inputs, using music to generate a sequence of movements, leveraging open source models to have a beautiful video that is consistent with the input.

1- **Learn how to create a Deep learning model**
Learn and understand a Deep learning model both through training a model from scratch and being able to use SOTA models in our project.

2- **Generate a sequence of poses out of music**
Train a model to be able to generate a choreography out of an audio file.

3- **Consistency**
Generate frames that are consistent on time and motion with the music.

4- **Beautiful output**
Besides the video itself, our intention is to make it beautiful to the viewer.


## Dataset

The data we used to train and validate our model was the AIST++ 
dataset as it has the best quality human dance motion data. 
Strengths: Detailed motion capture, with diverse dance styles.
Weakness: Does not generalize, as it is limited to dance motions.

It has the following characteristics:
- 1.4k Dance motion sequences
- 5 Hours of dance motion
- 10 Different dance genres

We use a ControlNet model in order to extract poses from the dataset’s frames. 
These extracted poses are then used to train our model.
![Pose extraction](/images/pose_extractions.png)

## Music to poses


### GAN components
We trained a GAN model to generate poses from an audio input. 
The generator creates fake pose samples, while the discriminator evaluates whether the poses are real or fake. 
Through an adversarial process, both networks improve, allowing the generator to produce better poses.

The video input is decomposed into a series of frames and a series of spectrograms that correspond to each frame. In order to generate a pose the generator takes a spectrogram and the previous pose in order to make an inference and predict the next pose.

The discriminator is solely dedicated to identifying false poses.

Now we delve into each of the components of the full GAN pix to pix style architecture.

#### Generator
The UNetGenerator is a neural network model based on the U-Net architecture.
This model integrates several components as residual blocks, dilated convolutions, self-attention, spatial attention, and ConvLSTM to enhance its capabilities. 
Here's a breakdown of its components and how they work together:

- **Down Blocks**:

	A series of convolutional layers (with optional batch normalization) followed by LeakyReLU activation, used to 
downsample the input image.

- **Up Blocks**:

	A series of convolutional layers followed by pixel shuffle for upsampling, batch normalization, dropout, and LeakyReLU activation.

- **Spectrogram Processor**:

	Processes spectrogram input using a separate module (not defined in the provided code).

- **ConvLSTM Cell**:

	Adds temporal dependencies to the model, enabling it to handle sequential data.

- **Self-Attention and Spatial Attention**:

	Adds attention mechanisms to focus on relevant parts of the input, enhancing the model's ability to capture important features.

- **Residual Blocks**:

	Adds residual connections to help with gradient flow and model training as part of the modified Unet.

- **Dilated Convolution Blocks**:

	Uses dilated convolutions to capture a wider range of pixels without increasing the number of parameters.

- **Final Output**:

	Produces the final output image using a series of convolutional layers and a Tanh activation function.

#### Discriminator

The MultiScaleDiscriminator evaluates the quality of generated images. 
This model employs multiple discriminators at different scales to capture both fine and coarse details, 
ensuring a more comprehensive assessment of the generated images.

This includes the concatenation of the real/generated image and any additional information like conditioning data (spectrograms).

- **PatchGAN Discriminators**:

	The model consists of two PatchGAN discriminators (disc1 and disc2), which are designed to classify whether overlapping 
image patches are real or fake.  The input x (which includes the real/generated image and any additional information) is passed to the first PatchGAN 
discriminator (disc1).

- **Downsampling**:

	The input x is also downsampled using average pooling (F.avg_pool2d) before being passed to the second PatchGAN 
    discriminator (disc2). The average pooling reduces the size of the input, enabling the second discriminator to focus 
    on a different scale of features.

- **Output**:

	The outputs from both discriminators (x1 and x2) are returned as a list. These outputs represent the patch-level predictions from both scales.

![Trainig diagram](/images/trainig_diagram.png)
![Inference diagram](/images/inference_diagram.png)

#### Losses
##### Losses from the Generator

- **Loss feature matching**

	The feature matching loss involves using the intermediate feature maps from the discriminator as a basis for 
comparison. Specifically, the generator's output is passed through the discriminator, and the features extracted 
from different layers are compared to those extracted from the real images. This encourages the generator to 
produce images with similar internal structures to real images.

- **Loss GAN**

	The GAN loss is the Binary Cross-Entropy (BCE) loss used in the GAN architecture. It measures how well the generator can fool the discriminator. The generator aims to minimize this loss to generate images that the discriminator classifies as real.

- **Loss L1**

	The L1 loss, also known as mean absolute error (MAE), compares the generated image with the real image on a pixel-by-pixel basis. This loss is often weighted to emphasize non-black pixels more than others, which can help the generator focus on critical regions of the image. 

- **Loss perceptual**

	The perceptual loss loads a pre-trained VGG network to evaluate the perceptual similarity between the generated and real images. Instead of focusing solely on pixel-wise differences, this loss penalizes discrepancies in the high-level features, capturing human perceptual differences better. 

##### Losses form the Discriminator
- **Loss real & loss fake**:

	The real and fake loss refers to the discriminator's ability to distinguish between real and fake images. The discriminator is trained to maximize the likelihood of correctly classifying real images as real and fake images as fake, often using BCE loss for this purpose.

- **Loss criterion gradient**

	The gradient penalty loss involves interpolation between real and fake images to enforce the Lipschitz constraint, which helps stabilize the training of the discriminator. This loss penalizes the gradient norm deviation from 1. 

#### Hyperparameter Tuning
We experimented with various hyperparameters to optimize the model performance. Here are the results of our tuning efforts:

##### Learning Rate
We started with an initial learning rate from 0.0005 and then moved to a higher value 0.001.
ALso during each epoch, the learning rate for the generator decreases. We have experimented with different values. It was initially divided by 10, then halved, and finally it is currently multiplied by 0.8 to fine-tune the adjustment dynamically.

##### Number of Epochs (5, 10, 15, and 20)
Lower learning rates and shorter runs (up to 10 epochs) initially we observed a stabilized loss. 
However, with longer runs (up to 20 epochs), we observed that the loss started to fluctuate as if in a true competition between generator and discriminator.
Also, we discovered that when increasing the learning rate with shorter epochs (5 or 10) we saw a similar fluctuation without needing to train for as many epochs.

##### Increasing Discriminator Layers
Coinciding with the increased learning rate, adding more layers to the discriminator showed more GAN-like competition behavior.
We increased the convolutional layers from one to five, enhancing the model's ability to capture complex features through more trainable parameters.

##### Architectural Changes
Replacing ReLU with LeakyReLU activation improved performance by allowing a small, non-zero gradient when the unit is not active.

##### Batch Size
Increasing the batch size accelerated the training process but did not significantly affect the final results.

##### Pixel Weighted Loss
Incorporating pixel-weighted loss improved image quality. The loss initially dropped slowly because the model had to learn from black regions (poses have significant black areas).
The loss value was divided by 10 when the pixel is black, resulting in a lower but more effective loss curve, enhancing the learning process and generating better images.

#### Pixel Weighted Loss
This section provides a comprehensive overview of the different loss functions used in the GAN architecture and the results of hyperparameter tuning, offering insights into how these elements contribute to the model's performance and training dynamics.

## SD Inference: Poses to Video

### Image and video generation: Main components

Stable Diffusion (SD) is generative artificial intelligence model that creates images from text and image prompts. 
We used one of Stable Diffusion's models in order to generate beautiful visual outputs. As an open source model, researchers, 
engineers and artists have made additional contributions, which have led to the augmentation of its initial capabilies.

Along with SD, we have leveraged AnimateDiff and ControlNet, which have been integrated together through a custom pipeline.
This custom pipeline, developed by the open source community, is a key part of the video generation.

In this section, we describe the different pieces that we have leveraged in order to generate a video from poses and a text prompt.

#### AnimateDiff SD model

The AnimateDiff SD model is a variant of the Stable Diffusion model specifically designed for generating animations.
It appends a motion modeling module to the frozen SD model and trains it on video clips in order to learn a motion prior.
The module is then used to generate animated images.

#### ControlNet

ControlNet is a neural network designed to control the generation process. It guides image generation using auxiliary inputs
like segmentation maps, depth maps, and edge maps. In our case, we use a pretrained ControlNet with an Openpose detector.

#### AnimateDiff ControlNet pipeline

In order to both generate a video from a prompt, maintaining pose consistency, we needed to combine an image generation model
with animation and pose control.

We found a pipeline that makes a connection between AnimateDiff and ControlNet. 
The default generation has 16 frames. In order to generate longer videos, we use 32 frames.
In addition to this, we it uses a motion pretrained motion adapter and a Variation Autoencoder, and can be used with other SD 1.5 models.

### Final results and conclusions

The integration of AnimateDiff SD model, ControlNet, and AnimateDiff resulted in the successful generation of high-quality videos from pose sequences. 
Here are some key observations and conclusions from our experiments:

**High-Quality Video Generation:** 

The combination of these elements allowed us to generate videos with high visual fidelity and smooth transitions between frames.

**Accurate Pose Adherence:** 

Poses ControlNet effectively guided the generation process, ensuring that the generated frames accurately followed the input pose sequences.

**Temporal Consistency:** 

TemporalNet significantly improved the temporal consistency of the generated videos, reducing flickering and abrupt transitions.

**Efficient Training:** 

The model components were optimized to work efficiently together, allowing for relatively fast inference times.
Overall, the integration of these components in the Stable Diffusion model framework demonstrated a powerful approach to video generation from poses.


## End to end pipeline

### Platform architecture
We build our application in python version 3.12 with the libraries specified in the requirements.txt file.
The project was developed in a Google Cloud VM with the following specifications:

And locally we used a conda environment.

The project is organized as a repository having the next components:

| File                                                                                                                                            | Description                                                                                                                                              |
|-------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| [data/](https://github.com/mpsanchezg/generative-art-project/tree/main/data)                                               | _Contains data in raw format._                                                                                                                           |
| [src/config.py](https://github.com/mpsanchezg/generative-art-project/blob/main/src/config.py)                                                                                                                                 | _Contains the main configuration of the project, global variables and paths_                                                                             |
| [src/custom_transformation.py](https://github.com/mpsanchezg/generative-art-project/blob/main/src/custom_transformation.py)                                                                                                                  | _Contains functions created ofr specific transformations used in train._                                                                                 |
| [src/extract_frames.py]                                                                                                                         | _Contains the classes to transform the video into frames and saving the to the data folder with specific names._                                         |
| [src/model.py](https://github.com/mpsanchezg/generative-art-project/blob/main/src/model.py)                                | Contains all the classes used in training, this means both Generator and Discriminator classes as well fo all the rest of the models used for training.  |
| [src/predict.py](https://github.com/mpsanchezg/generative-art-project/blob/main/src/predict.py)                   | all the classes used in training, this means the Stable diffusion model with its Controlnet used for inference.                                          |
| [src/train.py](https://github.com/mpsanchezg/generative-art-project/blob/main/src/traina.py)     | Executes the training process and saves best accuracy model parameters.                                                                                  |
| [src/utils.py](https://github.com/mpsanchezg/generative-art-project/blob/main/src/utils.py)     | Contains the functions used across the project.                                                                                                          |


## How to
### Install the project
You should create a Dockerfile and build the image using `docker build`
 
#### Install requirements
To install the requirements, you can run the command `pip install -r requirements.txt`.

#### Install conda
To install conda, you can follow the instructions in the [official documentation](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

#### Create your conda environment
```
conda create generative-art-project
```

#### Setting the environment in Google Drive
To set the environment in Google Drive, you can follow the instructions in the [official documentation](https://cloud.google.com/sdk/gcloud/reference/compute/ssh).
```
gcloud compute ssh --zone "asia-northeast3-b" "generative-art-project-vm" --project "aidl2024-generatie-art-project"
```

### Run the project


#### Running training-inference scripts locally with conda

```
pip install -r requirements
```

for training, you should run the following command

```
python {ROOT_DIR}/src/train.py 
```

To do prediction to make poses, you should run the following command

```
python {ROOT_DIR}/src/prediction.py 
```


To do inference, you should run the following command

```
python {ROOT_DIR}/src/inference_sd.py 
```



