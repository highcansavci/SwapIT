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

def generate_video(dst_frames_path, model_name="SwapIt", saved_models_dir="saved_model"):
    model_path = Path(saved_models_dir) / f'{model_name}.pth'
    dst_frames_path_ = Path(dst_frames_path)
    image = Image.open(next(dst_frames_path_.glob("*.jpg")))
    image_size = image.size
    result_video = cv2.VideoWriter(str(dst_frames_path_.parent / "fake.mp4"), cv2.VideoWriter_fourcc(*"MJPG"), 30, image.size)

    face_extractor = FaceExtractor(image_size, config_args)
    face_masker = FaceMasking(config_args)

    device = config_args.config["device"]
    config_image_size = int(config_args.config["model"]["image_shape"])
    encoder = Encoder().to(device)
    inter = Inter().to(device)
    decoder = Decoder().to(device)

    saved_model = torch.load(model_path, map_location=torch.device(device))
    encoder.load_state_dict(saved_model['encoder'])
    inter.load_state_dict(saved_model['inter'])
    decoder.load_state_dict(saved_model['decoder_src'])

    model = torch.nn.Sequential(encoder, inter, decoder)

    frames_list = sorted(dst_frames_path_.glob("*.jpg"))
    for idx, frame_path in enumerate(frames_list, 1):
        print(f"Working on {idx}/{len(frames_list)}")
        frame = cv2.imread(str(frame_path))
        retval, face = face_extractor.detect(frame)
        if face is None:
            result_video.write(frame)
            continue
        face_image, face = face_extractor.extract(frame, face[0])
        face_image = face_image[..., ::-1].copy()
        face_image_cropped = cv2.resize(face_image, (config_image_size, config_image_size))
        fic_torch = torch.tensor(face_image_cropped / 255., dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
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
            continue

        fake_face_image = cv2.resize(output_face, (face_image.shape[1], face_image.shape[0]))
        fake_face_image = fake_face_image[..., ::-1]
        frame[face[1]:face[1]+face[3], face[0]:face[0]+face[2]] = fake_face_image
        result_video.write(frame)

    result_video.release()