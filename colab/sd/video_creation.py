# Generate the video from the images
import os
import cv2

def create_video(image_folder, video_path):
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    images = sorted([img for img in os.listdir(image_folder) if img.endswith(".png")])
    print(f"Found {len(images)} images to compile into a video.")

    if images:
        first_image_path = os.path.join(image_folder, images[0])
        frame = cv2.imread(first_image_path)

        if frame is not None:
            height, width, layers = frame.shape
            print(f"Image dimensions: {width}x{height}")

            video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'DIVX'), 10, (width, height)) # Check what should be the frame rate of the video

            for image in images:
                img_path = os.path.join(image_folder, image)
                frame = cv2.imread(img_path)
                if frame is not None:
                    video.write(frame)
                else:
                    print(f"Warning: Skipping frame {img_path} because it could not be read.")

            cv2.destroyAllWindows()
            video.release()
            print(f"Video saved at {video_path}")
        else:
            print(f"Error: The first image at {first_image_path} could not be read.")
    else:
        print("No images found in the specified folder.")