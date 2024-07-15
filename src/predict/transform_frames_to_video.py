import os
import cv2
import pathlib

# define paths
current_p = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = current_p.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
print(DATA_DIR)

def transform_frames_in_video(video_filename, frames_dir):
    # Path to save the video
    video_path = os.path.join(DATA_DIR, f"{video_filename}.mp4")

    # Frames per second
    fps = 30

    # Get the list of all frame files and sort them numerically
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')],
                        key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Read the first frame to get the dimensions
    first_frame = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    height, width, layers = first_frame.shape

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    # Write each frame to the video
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        video.write(frame)

    # Release the VideoWriter object
    cv2.destroyAllWindows()
    video.release()

    print(f'Video saved at {video_path}')

if __name__ == "__main__":
    transform_frames_in_video("result1", "/Users/mapisangut/Documents/UPC/project/Generative Art Project/data/results/individual_frames")
