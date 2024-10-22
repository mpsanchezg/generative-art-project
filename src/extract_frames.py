import os

import cv2
import librosa
import numpy as np
from moviepy.editor import VideoFileClip

from src.config import DATA_DIR


def extract_frames(number_of_videos):
    # Change directory to the folder containing the videos

    # Get a list of all .mp4 files in the folder
    video_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
    print(video_files)
    if number_of_videos:
        video_files = video_files[:number_of_videos]

    # Create the 'frames' directory if it doesn't exist
    if not os.path.exists('frames'):
        os.makedirs('frames')
    # Loop through each video file
    for video_file in video_files:
        # Load the video
        cap = cv2.VideoCapture(video_file)

        # Initialize frame and spectrogram lists
        frames = []
        spectrograms = []

        # Get the total duration of the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = total_frames / fps
        # TODO how to get the audio from the video
        audio_path = video_file.replace('.mp4', '.wav')

        # Load the video file
        video = VideoFileClip(video_file)

        # Extract the audio and save it as a WAV file
        video.audio.write_audiofile(audio_path, codec='pcm_s16le')

        # Extract frames and spectrograms
        frame_time = 0  # initialize frame time to 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame to 128x128
            frame = cv2.resize(frame, (256, 256))

            # Extract audio and convert to spectrogram
            # Load a short segment of audio centered around the current frame

            # Pad the audio appropriately for the first and last frames
            if frame_time < 0.5:
                # For the first frames, load audio from the start and pad the beginning
                padding_duration = 0.5 - frame_time
                audio_file = '/home/eduardo/projects/generative-art-project/data/raining/audio.wav'
                y, sr = librosa.load(audio_file, sr=None, offset=0, duration=frame_time + 0.5)
                y_padded = np.pad(y, (int(sr * padding_duration), 0), 'constant')
            elif frame_time > total_duration - 0.5:
                # For the last frames, load audio from the end and pad the end
                padding_duration = 0.5 - (total_duration - frame_time)
                audio_offset = frame_time - 0.5
                y, sr = librosa.load(audio_path, sr=None, offset=audio_offset, duration=1 - padding_duration)
                y_padded = np.pad(y, (0, int(sr * padding_duration)), 'constant')
            else:
                # For all other frames, load 1 second of audio as before
                y, sr = librosa.load(audio_path, sr=None, offset=frame_time - 0.5, duration=1)
                y_padded = y

            win_length = 256  # window length in samples
            hop_length = 64  # hop length in samples
            D = librosa.amplitude_to_db(np.abs(librosa.stft(y_padded, win_length=win_length, hop_length=hop_length)),
                                        ref=np.max)

            # Resize spectrogram to 128x128
            D = cv2.resize(D, (256, 256))

            spectrograms.append(D)

            # Save frame and spectrogram to the 'frames' directory using the frame time as the filename
            frame_filename = 'frames/{}_{:.3f}_frame.npy'.format(video_file.split('.')[0], frame_time)
            spectrogram_filename = 'frames/{}_{:.3f}_spectrogram.npy'.format(video_file.split('.')[0], frame_time)
            np.save(frame_filename, frame)
            np.save(spectrogram_filename, spectrograms[-1])

            print(f'Saved {frame_filename} and {spectrogram_filename}')

            frames.append(frame)
            frame_time += 1 / fps  # increment frame time by the duration of one frame

        cap.release()

        # Ensure the 'results' directory exists
        os.makedirs('results', exist_ok=True)
