# Generative Art Project

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


### Build Docker image

```
docker build -f Dockerfile -t gap_image .
```

### Train

```
docker run -v ${ROOT_DIR}/data:/data -it gap_image train
```
