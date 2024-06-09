
from controlnet_aux import OpenposeDetector
from diffusers.utils import load_image
from .ImageExtension import ImageExtension

class PretrainedOpenPoseModel:
    def __init__(
        pretrained_model: str = "lllyasviel/ControlNet",
    ):
        self.model = model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    def save_pose_from_frame(
        filename: str,
        frame_path: str,
        pose_path: str,
        image_extension: ImageExtension = ImageExtension.NPY
    ):
        # frame_filename_jpg = f"{filename}/images/{frame_time_rounded}_frame.jpg"
        # cv2.imwrite(frame_filename_jpg, frame)
        image = load_image(frame_path)

        pose = self.model(image)

        pose_extension = image_extension.name.lower()
        pose_filename = f"{pose_path}.{pose_extension}"
        if (pose_extension == "npy"):
            pose.save(pose_filename_jpg)
        else:
            pose.save(pose_filename)
        

