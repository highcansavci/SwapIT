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
                                               output_facial_transformation_matrixes=False,
                                               num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def get_mask(self, image):
        with mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.0,
            min_tracking_confidence=0.0
        ) as face_mesh:
            img = image.astype(np.uint8).copy()
            detection_result = face_mesh.process(img)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            if detection_result.multi_face_landmarks is None:
                print("No landmark detected.")
                return np.ones(image.shape[:2], dtype=np.uint8)
            x = np.array([landmark.x * image.shape[1] for landmark in detection_result.multi_face_landmarks[0].landmark], dtype=np.float32)
            y = np.array([landmark.y * image.shape[0] for landmark in detection_result.multi_face_landmarks[0].landmark], dtype=np.float32)

            hull = np.round(np.squeeze(cv2.convexHull(np.column_stack((x, y))))).astype(np.int32)

            mask = cv2.fillConvexPoly(mask, hull, 255)
            kernel = np.ones((7, 7), np.uint8)
            mask = cv2.erode(mask, kernel)

        return mask
