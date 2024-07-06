# Load the poses from Google Cloud Platform
import os
from glob import glob
import numpy as np
from google.cloud import storage

def load_pose_data_from_gcs(bucket_name, folder_path):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=folder_path)

    npy_files = []
    for blob in blobs:
        if blob.name.endswith('.npy'):
            npy_files.append(blob.name)

    if not npy_files:
        raise ValueError("No npy files found in the specified folder.")

    # Review with Laia if we should use a image format to pass the poses to the frozen SD model
    images = []
    for npy_file in npy_files:
        blob = bucket.blob(npy_file)
        npy_data = blob.download_as_string()
        image = np.load(io.BytesIO(npy_data))
        images.append(image)

    return images

# data_loading.py
import os
from glob import glob
import numpy as np

# Path to the folder containing the npy files
folder_path = '' # Add the Google Cloud path where images are stored

def load_pose_data(folder_path):
    # List all npy files in the folder
    npy_files = sorted(glob(os.path.join(folder_path, '*.npy')))

    # Check if npy_files is not empty
    if not npy_files:
        raise ValueError("No npy files found in the specified folder.")

    # Load the npy files as images
    # Review with Laia if we should use a image format to pass the poses to the frozen SD model
    images = [np.load(file) for file in npy_files]
    return images