from pathlib import Path
from PIL import Image
import numpy as np
import cv2
import torch
from face_extraction.tools import FaceExtractor
from face_extraction.masking import FaceMasking
from model.faceswap_model import Encoder, Inter, Decoder
from config.config import Config

config_args = Config()


def generate_camera(model_name="SwapIt", saved_models_dir="saved_model"):
    model_path = Path(saved_models_dir) / f'{model_name}.pth'

    face_extractor = FaceExtractor(640, config_args)
    face_masker = FaceMasking(config_args)

    device = config_args.config["device"]
    config_image_size = config_args.config["model"]["image_shape"]
    encoder = Encoder().to(device)
    inter = Inter().to(device)
    decoder = Decoder().to(device)

    saved_model = torch.load(model_path)
    encoder.load_state_dict(saved_model['encoder'])
    inter.load_state_dict(saved_model['inter'])
    decoder.load_state_dict(saved_model['decoder_src'])

    model = torch.nn.Sequential(encoder, inter, decoder)
    win_name = 'DeepFake'

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    while True:
        _, frame = cap.read()

        retval, face = face_extractor.detect(frame)
        if face is None:
            cv2.imshow(win_name, frame)
            continue
        face_image, face = face_extractor.extract(frame, face[0])
        face_image = face_image[..., ::-1].copy()
        face_image_cropped = cv2.resize(face_image, (config_image_size, config_image_size))
        fic_torch = torch.tensor(face_image_cropped / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(
            device)
        generated_face_torch = model(fic_torch)
        generated_face = (generated_face_torch.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

        mask_origin = face_masker.get_mask(face_image_cropped)
        mask_fake = face_masker.get_mask(generated_face)

        origin_moments = cv2.moments(mask_origin)
        cx = np.round(origin_moments['m10'] / origin_moments['m00']).astype(int)
        cy = np.round(origin_moments['m01'] / origin_moments['m00']).astype(int)

        try:
            output_face = cv2.seamlessClone(generated_face, face_image_cropped, mask_fake, (cx, cy), cv2.NORMAL_CLONE)
        except:
            print("Skip")
            cv2.imshow(win_name, frame)
            continue

        fake_face_image = cv2.resize(output_face, (face_image.shape[1], face_image.shape[0]))
        fake_face_image = fake_face_image[..., ::-1]
        frame[face[1]:face[1] + face[3], face[0]:face[0] + face[2]] = fake_face_image

        cv2.imshow(win_name, frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cv2.destroyAllWindows()
