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
  * [Architecture](#architecture)
	 * [GAN](#optical-flow)
	 * [SD](#i3d)
	 * [ControlNet](#classifier-neural-network)
  * [Generation of poeple dancign](#art_generation)
	 * [GAN improvements](#gan_improvements)
	 	* [Generator](#Generator)
	 	* [Discriminator](#rdiscriminator)
	 	* [Losses](#losses)
	 	* [Paper: ](#paper)
	 	* [Conclusions](#conclusions)
	 * [Model improvements](#model-improvements)
		* [First approach: ](#first-approach-rgb-videos)
		* [Second approach: ](#second-approach-optical-flow-videos)
		* [Third approach: ](#third-approach-two-stream-rgb-and-optical-flow-videos)
		* [Final results](#final-results)
	 * [Hyperparameter Tuning](#hyperparameter-tuning)
		* [...](#scheduler)
		* [...](#hyperoptsearch)
  * [End to end system](#end-to-end-system)
	 * [Platform architecture](#platform-architecture)
		* [Video capturing](#video-capturing)
		* [Video processing](#video-processing)
		* [Model](#model)
  * [How to](#how-to)
	 * [How to prepare the dataset from scratch](#how-to-prepare-the-dataset-from-scratch)
		* [How to transform videos to poses](#how-to-extract-features-from-videos)
		* [Clean dataset](#clean-dataset)
		* [Convert video frames to video format](#convert-video-frames-to-video-format)
	 * [How to train the model](#how-to-train-the-model)
		* [Setting the environment in Google Drive](#setting-the-environment-in-google-drive)
        * [Running training scripts locally with conda](#setting-the-environment-in-google-drive)
		* [Running training scripts locally with Docker](#setting-the-environment-in-google-drive)
        * [Running training scripts from VM in GCP](#setting-the-environment-in-google-drive)
	 * [How to create new video from a music file](#how-to-run-the-program---video_processor)
        * [Leaving the music file in Google Drive](#setting-the-environment-in-google-drive)
        * [Running inference scripts locally with conda](#setting-the-environment-in-google-drive)
		* [Running inference scripts locally with Docker](#setting-the-environment-in-google-drive)
        * [Running inference scripts from VM in GCP](#setting-the-environment-in-google-drive)
		* [Installation](#installation)
		   * [Install requirements](#install-requirements)
		   * [Install conda](#install-miniconda)
		   * [Create your conda environment](#create-your-miniconda-environment)
		 * [RUN the project](#run-the-project)
			  * [Video processor app](#video-processor-app)
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


- GAN
- SD

![Trainig diagram](/images/trainig_diagram.png)
![Inference diagram](/images/inference_diagram.png)

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


