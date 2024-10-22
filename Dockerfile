# set base image (host OS)
FROM python:3.8-slim

# set the working directory in the container
WORKDIR /app

# copy the dependencies file to the working directory
COPY requirements.txt requirements.txt

# install dependencies
RUN pip install --quiet -r requirements.txt

# Copy code to the working directory
COPY src/ /app

# command to run on container start
ENTRYPOINT ["python", "./entrypoint.py"]
