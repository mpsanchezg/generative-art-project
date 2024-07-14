import os
from model_loading import pipe
from data_loading import load_pose_data_from_gcs
from image_generation import generate_images
from video_creation import create_video

def main():
    # Set Google Cloud Storage bucket and folder path
    bucket_name = 'your-gcs-bucket-name'
    folder_path = 'path/to/poses'

    # Load pose data from Google Cloud Storage
    images = load_pose_data_from_gcs(bucket_name, folder_path)

    # Generate images
    save_dir = 'path/to/save/generated_images'
    generate_images(pipe, images, save_dir)

    # Create video from generated images
    video_path = 'path/to/save/generated_video/video.avi'
    create_video(save_dir, video_path)

if __name__ == "__main__":
    main()