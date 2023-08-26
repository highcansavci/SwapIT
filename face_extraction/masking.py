from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as python_mp
from mediapipe.tasks.python import vision


class FaceMasking:
    def __init__(self, config):
        landmarks_model_path = Path(config.config["landmark"]["model_path"])
        if not landmarks_model_path.exists():
            landmarks_model_path.parent.mkdir(parents=True, exist_ok=True)
            print('Downloading Face Landmarks Model...')
            filename, headers = urlretrieve(config.config["landmark"]["url_path"], filename=str(landmarks_model_path))
            print('Finished Downloading...')

        base_options = python_mp.BaseOptions(model_asset_path=str(landmarks_model_path))
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=False,
                                               output_facia_transformation_matrixes=False,
                                               num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def get_mask(self, image):
        im_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image.astype(np.uint8).copy())
        detection_result = self.detector.detect(im_mp)

        x = np.array([landmark.x * image.shape[1] for landmark in detection_result.face_landmarks[0]], dtype=np.float32)
        y = np.array([landmark.y * image.shape[0] for landmark in detection_result.face_landmarks[0]], dtype=np.float32)

        hull = np.round(np.squeeze(cv2.convexHull(mp.column_stack((x, y))))).astype(np.int32)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mask = cv2.fillConvexPoly(mask, hull, 255)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.erode(mask, kernel)

        return mask
