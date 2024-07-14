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
		* [...](#learning-rate)
        * [...](#batch-size)
        * [...]
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
These pose extractions are then used to train our model.
![Pose extraction](/images/pose_extractions.png)

## Architecture

We build our application in python version 3.12 with the libraries specified in the requirements.txt file.
The project was developed in a Google Cloud VM with the following specifications:

And locally we used a conda environment.

The project is organized as a repository having the next components:

| File                                                                                                                                            | Description                                                                                                                                              |
|-------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| [data/](https://github.com/gesturesAidl/video_processor/blob/main/scripts/training/dataset.ipynb)                                               | _Contains data in raw format._                                                                                                                           |
| [src/config.py]                                                                                                                                 | _Contains the main configuration of the project, global variables and paths_                                                                             |
| [src/custom_transformation.py]                                                                                                                  | _Contains functions created ofr specific transformations used in train._                                                                                 |
| [src/extract_frames.py]                                                                                                                         | _Contains the classes to transform the video into frames and saving the to the data folder with specific names._                                         |
| [src/model.py](https://github.com/gesturesAidl/video_processor/blob/main/scripts/training/main_two_stream.ipynb)                                | Contains all the classes used in training, this means both Generator and Discriminator classes as well fo all the rest of the models used for training.  |
| [src/predict.py](https://github.com/gesturesAidl/video_processor/blob/main/scripts/training/main_two_stream_OneCycleLR.ipynb)                   | all the classes used in training, this means the Stable diffusion model with its Controlnet used for inference.                                          |
| [src/train.py](https://github.com/gesturesAidl/video_processor/blob/main/scripts/training/main_two_stream_OneCycleLR_save_best_model.ipynb)     | Executes the training process and saves best accuracy model parameters.                                                                                  |
| [src/utils.py](https://github.com/gesturesAidl/video_processor/blob/main/scripts/training/main_two_stream_OneCycleLR_save_best_model.ipynb)     | Contains the functions used across the project.                                                                                                          |


### GAN components
We trained a GAN model to generate from an audio input to poses. 
The generator creates fake poses samples, while the discriminator evaluates whether the poses are real or fake. 
Through an adversarial process, both networks improve, allowing the generator to produce better poses.

The video input is decomposed by a series of frames and a series of spectrograms. In order to generate a pose to "dance" is to show to the 
generator a spectrogram and the previous pose in order to make an inference over the next pose.

The discriminator is solely dedicated to identify false poses.

Now we delve into each of the components of the full GAN pix to pix style architecture.

#### Generator
The UNetGenerator is a neural network model based on the U-Net architecture.
This model integrates several components as residual blocks, dilated convolutions, self-attention, spatial attention, and ConvLSTM to enhance its capabilities. 
Here's a breakdown of its components and how they work together:

- **Down Blocks**:

	A series of convolutional layers (with optional batch normalization) followed by LeakyReLU activation, used to 
downsample the input image.

- **Up Blocks**:

	A series of convolutional layers followed by pixel shuffle for upsampling, batch normalization, dropout (optional), and LeakyReLU activation.

- **Spectrogram Processor**:

	Processes spectrogram input using a separate module (not defined in the provided code).

- **ConvLSTM Cell**:

	Adds temporal dependencies to the model, enabling it to handle sequential data.

- **Self-Attention and Spatial Attention**:

	Adds attention mechanisms to focus on relevant parts of the input, enhancing the model's ability to capture important features.

- **Residual Blocks**:

	Adds residual connections to help with gradient flow and model training.

- **Dilated Convolution Blocks**:

	Uses dilated convolutions to capture a wider range of context without increasing the number of parameters.

- **Final Output**:

	Produces the final output image using a series of convolutional layers and a Tanh activation function.

#### Discriminator

The MultiScaleDiscriminator evaluates the quality of generated images. 
This model employs multiple discriminators at different scales to capture both fine and coarse details, 
ensuring a more comprehensive assessment of the generated images.

This includes the concatenation of the real/generated image and any additional information like conditioning data.

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

	Penalizes # TODO: complete 

- **Loss GAN**

	The BCE loss form the GAN architecture # TODO: complete 

- **Loss L1**

	# TODO: complete 

- **Loss perceptual**

	Loads a VGG model that is trained to penalize over lack of human perception as an image #TODO: complete 

##### Losses form the Discriminator
- **Loss real & loss fake**:

	# TODO: complete 

- **Loss criterion gradient**

	# TODO: complete 

### SD components



## Evaluation

Evaluating dancing could be subjective and highly dependent on cultural aspects.

We will use three evaluation methods, in which we balance automated metrics and subjective evaluations in order to assess the quality of our outputs.

Ground truth poses vs. model poses
We calculate the distributional spread of generated dances compared to ground truth dances. We measure the spread in the kinetic (related to movement) and geometric (related to shapes) feature spaces.
Our goal would be to emulate our training data, therefore we would target having scores that match the score of ground truth distribution.



## Installation
You should create a Dockerfile and build the image using `docker build`
 
## Running the project
Once the project is done, you can train it by running the command `docker run <IMAGE_NAME> train`, and predict with the command `docker run <IMAGE_NAME> predict <INPUT_FEATURES>`. Note that you will need to mount some volumes when using `docker run`, otherwise these commands won't work.

## Run

### Environment setup

```
conda create generative-art-project
```

```
conda activate generative-art-project
```

```
pip install -r requirements
```

## Connect to Google VM

```
gcloud compute ssh --zone "asia-northeast3-b" "generative-art-project-vm" --project "aidl2024-generatie-art-project"
```

### Build Docker image

```
docker build -f Dockerfile -t gap_image .
```

### Train

```
docker run -v ${ROOT_DIR}/data:/data -it gap_image train
```


### Inferecne

```
docker run -v ${ROOT_DIR}/data:/data -it gap_image inference
```


