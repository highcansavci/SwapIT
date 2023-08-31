import sys
import os
import datetime
import time
from random import choice
from glob import glob
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from config.config import Config

config_args = Config()

def get_training_data(images, batch_size):
    indices = np.random.randint(len(images), size=batch_size)
    for i, index in enumerate(indices):
        image = images[index]
        image = random_transform(image, **config_args.config["transform_args"])
        warped_img, target_img = random_warp(image)

        if i == 0:
            warped_images = np.empty((batch_size,) + warped_img.shape, warped_img.dtype)
            target_images = np.empty((batch_size,) + target_img.shape, warped_img.dtype)

        warped_images[i] = warped_img
        target_images[i] = target_img

    return warped_images, target_images


def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    height, width = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * width
    ty = np.random.uniform(-shift_range, shift_range) * height
    matrix = cv2.getRotationMatrix2D((width // 2, height // 2), rotation, scale)
    matrix[:, 2] = (tx, ty)
    result = cv2.warpAffine(image, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    flip_prob = np.random.random()
    if flip_prob < random_flip:
        result = result[:, ::-1]
    return result


def random_warp(image):
    height, width = image.shape[:2]
    range_ = np.linspace(height * 0.1, height * 0.9, 5)
    map_x = np.broadcast_to(range_, (5, 5))
    map_y = map_x.T

    map_x = map_x + np.random.normal(size=(5, 5), scale=5 * height / 256)
    map_y = map_y + np.random.normal(size=(5, 5), scale=5 * height / 256)

    interp_map_x = cv2.resize(map_x, (int(width / 2 * (1 + 0.25)), int(height / 2 * (1 + 0.25))))[
                   int(width / 2 * 0.25 / 2):int(width / 2 * (1 + 0.25) - width / 2 * 0.25 / 2),
                   int(width / 2 * 0.25 / 2):int(width / 2 * (1 + 0.25) - width / 2 * 0.25 / 2)].astype('float32')
    interp_map_y = cv2.resize(map_y, (int(width / 2 * (1 + 0.25)), int(height / 2 * (1 + 0.25))))[
                   int(width / 2 * 0.25 / 2):int(width / 2 * (1 + 0.25) - width / 2 * 0.25 / 2),
                   int(width / 2 * 0.25 / 2):int(width / 2 * (1 + 0.25) - width / 2 * 0.25 / 2)].astype('float32')

    warped_image = cv2.remap(image, interp_map_x, interp_map_y, cv2.INTER_LINEAR)

    src_points = np.stack([map_x.ravel(), map_y.ravel()], axis=-1)
    dst_points = np.mgrid[0:width // 2 + 1:width // 8, 0:height // 2 + 1:height // 8].T.reshape(-1, 2)

    A = np.zeros((2 * src_points.shape[0], 2))
    A[0::2, :] = src_points
    A[0::2, 1] = -A[0::2, 1]
    A[1::2, :] = src_points[:, ::-1]
    A = np.hstack((A, np.tile(np.eye(2), (src_points.shape[0], 1))))
    b = dst_points.flatten()

    similarity_matrix = np.linalg.lstsq(A, b, rcond=None)[0]
    similarity_matrix = np.array([[similarity_matrix[0], -similarity_matrix[1], similarity_matrix[2]],
                                  [similarity_matrix[1], similarity_matrix[0], similarity_matrix[3]]])

    target_image = cv2.warpAffine(image, similarity_matrix, (width // 2, height // 2))

    return warped_image, target_image


class FaceDataset(Dataset):
    def __init__(self, data_path):
        self.image_files_src = glob(data_path + "/src/aligned/*.jpg")
        self.image_files_dst = glob(data_path + "/dst/aligned/*.jpg")

    def __len__(self):
        return min(len(self.image_files_src), len(self.image_files_dst))

    def __getitem__(self, idx):
        img_shape = int(config_args.config["model"]["image_shape"])
        img_file_src = choice(self.image_files_src)
        img_file_dst = choice(self.image_files_dst)
        img_src = np.asarray(Image.open(img_file_src).resize((2 * img_shape, 2 * img_shape)))
        img_dst = np.asarray(Image.open(img_file_dst).resize((2 * img_shape, 2 * img_shape)))

        return img_src, img_dst

    def collate_fn(self, batch):
        device = config_args.config["device"]
        images_src, images_dst = list(zip(*batch))
        warp_image_src, target_image_src = get_training_data(images_src, len(images_src))
        warp_image_src = torch.tensor(warp_image_src, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        target_image_src = torch.tensor(target_image_src, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        warp_image_dst, target_image_dst = get_training_data(images_dst, len(images_dst))
        warp_image_dst = torch.tensor(warp_image_dst, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        target_image_dst = torch.tensor(target_image_dst, dtype=torch.float32).permute(0, 3, 1, 2).to(device)

        return warp_image_src, target_image_src, warp_image_dst, target_image_dst
