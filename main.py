from pathlib import Path
import face_extraction.tools as fet
import model.faceswap_model as fs
from inference.generate_video import generate_video
from inference.generate_camera import generate_camera
from config.config import Config

config = Config()

extract_and_align_src = config.config["main"]["extract_and_align_src"]
extract_and_align_dst = config.config["main"]["extract_and_align_dst"]
train = config.config["main"]["train"]
eval_video = config.config["main"]["eval_video"]
eval_camera = config.config["main"]["eval_camera"]

model_name = config.config["main"]["model_name"]
new_model = not config.config["main"]["checkpoint"]

data_root = Path(config.config["main"]["data_root"])
src_video_path = data_root / 'data_src.mp4'
dst_video_path = data_root / 'data_dst.mp4'

src_processing_folder = data_root / 'src'
dst_processing_folder = data_root / 'dst'

if extract_and_align_src:
    fet.extract_frames_from_video(video_path=src_video_path, output_folder=src_processing_folder, frames_to_skip=0)
if extract_and_align_dst:
    fet.extract_frames_from_video(video_path=dst_video_path, output_folder=dst_processing_folder, frames_to_skip=0)

if extract_and_align_src:
    fet.extract_align_face_from_img(input_dir=src_processing_folder, desired_face_width=256)
if extract_and_align_dst:
    fet.extract_align_face_from_img(input_dir=dst_processing_folder, desired_face_width=256)

if train:
    fs.train(str(data_root), model_name, new_model, saved_models_dir='saved_model')

if eval_video:
    generate_video(dst_processing_folder, model_name, saved_models_dir='saved_model')

if eval_camera:
    generate_camera(model_name, saved_models_dir='saved_model')