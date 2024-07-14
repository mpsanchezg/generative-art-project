import os
import pathlib

# define paths
PROJECT_NAME = "generative-art-project"

current_p = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = current_p.parent
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
GENERATED_FRAMES_DIR = os.path.join(RESULTS_DIR, "generated_frames")
OUTPUT_VIDEO_DIR = os.path.join(RESULTS_DIR, "generated-video")

