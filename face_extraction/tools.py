import cv2
from pathlib import Path
from shutil import rmtree
from urllib.request import urlretrieve
import numpy as np
from tqdm import tqdm
from typing import Union
import os, random


def extract_frames_from_video(video_path: Union[str, Path], output_folder: Union[str, Path],
                              frames_to_skip: int = 0) -> None:
    video_path_ = Path(video_path)
    output_folder_ = Path(output_folder)

    if not video_path_.exists():
        raise ValueError(f"The path to the video file {video_path_.absolute()}")
    if not output_folder_.exists():
        output_folder_.mkdir(parents=True)

    video_capture = cv2.VideoCapture(str(video_path))

    extract_frame_counter = 0
    saved_frame_counter = 0

    while True:
        _, frame = video_capture.read()
        if not _:
            break

        if extract_frame_counter % (frames_to_skip + 1) == 0:
            cv2.imwrite(str(output_folder_ / f"{saved_frame_counter:05d}.jpg"))
            saved_frame_counter += 1

        extract_frame_counter += 1

        print(f"{saved_frame_counter} of {extract_frame_counter} frames saved successfully.")
        video_capture.release()
        cv2.destroyAllWindows()


def extract_align_face_from_img(input_dir: Union[str, Path], desired_face_width: int = 256) -> None:
    input_dir_ = Path(input_dir)
    output_dir_ = input_dir_ / "aligned"
    if output_dir_.exists():
        rmtree(output_dir_)
    output_dir_.mkdir()

    random_image_path = random.choice(
        [x for x in os.listdir(input_dir_) if os.path.isfile(os.path.join(input_dir_, x))])
    image = cv2.imread(str(random_image_path))
    image_height = image.shape[0]
    image_width = image.shape[1]

    detector = FaceExtractor((image_width, image_height))

    for image_path in tqdm(list(input_dir_.glob("*.jpg"))):
        image = cv2.imread(str(image_path))

        ret, faces = detector.detect(image)
        if faces is None:
            continue

        face_aligned = detector.align(image, faces[0, :], desired_face_width)
        cv2.imwrite(str(output_dir_ / f"{image_path.name}"), face_aligned, [cv2.IMWRITE_JPEG_QUALITY, 90])


class FaceExtractor:
    def __init__(self, image_size, config):
        detection_model_path = Path(config.config["detection"]["model_path"])
        if not detection_model_path.exists():
            detection_model_path.parent.mkdir(parents=True, exist_ok=True)
            url = config.config["detection"]["url_path"]
            print('Downloading Face Detection Model...')
            filename, headers = urlretrieve(url, filename=str(detection_model_path))
            print("Finished Downloading...")

        self.detector = cv2.FaceDetectorYN.create(str(detection_model_path), "", image_size)

    def detect(self, image):
        ret, faces = self.detector.detect(image)
        return ret, faces

    @staticmethod
    def align(image, face, desired_face_width=256, left_eye_desired_coord=np.array((0.37, 0.37))):
        desired_face_height = desired_face_width
        right_eye_desired_coord = np.array((1 - left_eye_desired_coord[0], left_eye_desired_coord[1]))

        right_eye = face[4:6]
        left_eye = face[6:8]

        dist_eyes_x = right_eye[0] - left_eye[0]
        dist_eyes_y = right_eye[1] - left_eye[1]
        dist_between_eyes = np.sqrt(dist_eyes_x ** 2 + dist_eyes_y ** 2)
        angles_between_eyes = np.rad2deg(np.arctan2(dist_eyes_y, dist_eyes_x) - np.pi)
        eyes_center = (left_eye + right_eye) // 2

        desired_dist_between_eyes = desired_face_width * (right_eye_desired_coord[0] - left_eye_desired_coord[0])
        scale = desired_dist_between_eyes / dist_between_eyes

        matrix = cv2.getRotationMatrix2D(eyes_center, angles_between_eyes, scale)
        matrix[0, 2] += 0.5 * desired_face_width - eyes_center[0]
        matrix[1, 2] += left_eye_desired_coord[1] * desired_face_height - eyes_center[1]

        face_aligned = cv2.warpAffine(image, matrix, (desired_face_width, desired_face_height), flags=cv2.INTER_CUBIC)
        return face_aligned

    @staticmethod
    def extract(image, face, desired_face_width=25, left_eye_desired_coord=np.array((0.37, 0.37))):
        desired_face_height = desired_face_width

        right_eye = face[4:6]
        left_eye = face[6:8]

        eyes_center = (left_eye + right_eye) // 2

        x_left = (eyes_center[0] - desired_face_width // 2).astype(int)
        width = desired_face_width
        y_top = (eyes_center[1] - left_eye_desired_coord[1] * desired_face_height).astype(int)
        height = desired_face_height
        face = image[y_top:y_top+height, x_left:x_left+width]
        return face, [x_left, y_top, width, height]



