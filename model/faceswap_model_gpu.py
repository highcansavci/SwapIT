from pathlib import Path
import numpy as np
import cv2
import torchvision.utils
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config.config import Config
from dataset.dataset import FaceDataset

config_args = Config()
plt.style.use("dark_background")
px = 1 / plt.rcParams["figure.dpi"]


def pixel_norm(x, dim=-1):
    return x / torch.sqrt(torch.mean(x ** 2, dim=dim, keepdim=True) + 1e-6)


def depth2space(x, size=2):
    batch_size, channels, height, width = x.shape
    out_height = size * height
    out_width = size * width
    out_channels = channels // (size * size)
    x = x.reshape((-1, size, size, out_channels, height, width))
    x = x.permute((0, 3, 4, 1, 5, 2))
    x = x.reshape((-1, out_channels, out_height, out_width))
    return x


class Depth2Space(nn.Module):
    def __init__(self):
        super(Depth2Space, self).__init__()

    def forward(self, x, size=2):
        return depth2space(x, size)


class WSConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(WSConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.scale = (2 / (in_channels * (kernel_size ** 2)) + 1e-6) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            WSConv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(64),
            WSConv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(128),
            WSConv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(256),
            WSConv2d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.1, True),
            nn.BatchNorm2d(512),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = pixel_norm(x, dim=-1)
        return x


class WSConv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding="same"):
        super(WSConv2dSame, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding="same")
        self.scale = (2 / (in_channels * (kernel_size ** 2)) + 1e-6) ** 0.5
        self.bias = self.conv.bias
        self.conv.bias = None

        nn.init.normal_(self.conv.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.conv(x * self.scale) + self.bias.view(1, self.bias.shape[0], 1, 1)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upsample = nn.Sequential(
            WSConv2dSame(in_channels, out_channels),
            nn.LeakyReLU(0.1, True),
            Depth2Space()
        )

    def forward(self, x):
        x = self.upsample(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = WSConv2dSame(in_channels, in_channels)
        self.conv2 = WSConv2dSame(in_channels, in_channels)

    def forward(self, input):
        x = self.conv1(input)
        x = nn.functional.leaky_relu(x, 0.2)
        x = self.conv2(x)
        x += input
        x = nn.functional.leaky_relu(x, 0.2)
        return x


class WSLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(WSLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = (2 / in_features + 1e-6) ** 0.5
        self.bias = self.linear.bias
        self.linear.bias = None

        nn.init.normal_(self.linear.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.linear(x * self.scale) + self.bias


class Inter(nn.Module):
    def __init__(self):
        super(Inter, self).__init__()
        self.inter = nn.Sequential(
            WSLinear(18432, 128),
            nn.BatchNorm1d(128),
            WSLinear(128, 1152),
            nn.BatchNorm1d(1152),
            nn.Unflatten(1, (128, 3, 3)),
            Upsample(128, 512)
        )

    def forward(self, x):
        x = self.inter(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            Upsample(128, 2048),
            ResBlock(512),
            nn.BatchNorm2d(512),
            Upsample(512, 1024),
            ResBlock(256),
            nn.BatchNorm2d(256),
            Upsample(256, 512),
            ResBlock(128),
            nn.BatchNorm2d(128),
        )
        self.conv_out = WSConv2dSame(128, 3, kernel_size=1)
        self.conv_out1 = WSConv2dSame(128, 3, kernel_size=3)
        self.conv_out2 = WSConv2dSame(128, 3, kernel_size=3)
        self.conv_out3 = WSConv2dSame(128, 3, kernel_size=3)
        self.depth_to_space = Depth2Space()

    def forward(self, x):
        x = self.decoder(x)
        out = self.conv_out(x)
        out1 = self.conv_out1(x)
        out2 = self.conv_out2(x)
        out3 = self.conv_out3(x)
        x = torch.concat((out, out1, out2, out3), 1)
        x = self.depth_to_space(x, 2)
        x = nn.functional.sigmoid(x)
        return x


def create_window(size=11, sigma=1.5, channels=1):
    # 2D Gaussian Window
    gaussian_kernel_1d = torch.tensor(cv2.getGaussianKernel(size, sigma), dtype=torch.float32)
    gaussian_kernel_2d = gaussian_kernel_1d @ gaussian_kernel_1d.t()
    gaussian_kernel_2d = gaussian_kernel_2d.expand((channels, 1, size, size)).contiguous().clone()
    return gaussian_kernel_2d


def sdsim(first_image, second_image, window_size=11):
    pad = window_size // 2
    window = create_window(window_size, channels=3).to(config_args.config["device"])
    eps = 1e-6

    mu1 = nn.functional.conv2d(first_image, window, padding=pad, groups=3)
    mu2 = nn.functional.conv2d(second_image, window, padding=pad, groups=3)

    mu1_squared = mu1 ** 2
    mu2_squared = mu2 ** 2
    mu_cov = mu1 * mu2

    sigma1_squared = nn.functional.conv2d(first_image * first_image, window, padding=pad, groups=3) - mu1_squared + eps
    sigma2_squared = nn.functional.conv2d(second_image * second_image, window, padding=pad,
                                          groups=3) - mu2_squared + eps
    sigma_cov = nn.functional.conv2d(first_image * second_image, window, padding=pad, groups=3) - mu_cov + eps

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    c3 = c2 / 2

    luminosity_metric = (2 * mu_cov + c1) / (mu1_squared + mu2_squared + c1)
    contrast_metric = (2 * torch.sqrt(sigma1_squared * sigma2_squared) + c2) / (sigma1_squared + sigma2_squared + c2)
    structure_metric = (sigma_cov + c3) / (torch.sqrt(sigma1_squared * sigma2_squared) + c3)

    ssim = luminosity_metric * contrast_metric * structure_metric
    dssim = (1 - ssim.mean()) / 2

    return dssim


def draw_results(reconstruct_src, target_src, reconstruct_dst, target_dst, fake, loss_src, loss_dst):
    fig, axes = plt.subplots(figsize=(660 * px, 370 * px))
    axes.plot(loss_src, label='loss src')
    axes.plot(loss_dst, label='loss dst')
    plt.legend()
    plt.title(f'press q to quit and save, or r to refresh\nepoch = {len(loss_src)}')
    canvas = fig.canvas
    canvas.draw()
    width, height = canvas.get_width_height()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    image_array = image_array.reshape((height, width, 3)) / 255.

    images_for_grid = []
    for ii in range(3):
        images_for_grid.extend([reconstruct_src[ii], target_src[ii], reconstruct_dst[ii], target_dst[ii], fake[ii]])

    im_grid = torchvision.utils.make_grid(images_for_grid, 5, padding=30).permute(1, 2, 0).cpu().numpy()
    final_image = np.vstack([image_array, im_grid])
    final_image = final_image[..., ::-1]  # convert to BGR
    return final_image


def train(data_path: str, model_name: "SwapIt", new_model=False, saved_models_dir="saved_model"):
    saved_models_dir_ = Path(saved_models_dir)
    learning_rate = float(config_args.config["model"]["learning_rate"])
    device = config_args.config["device"]
    batch_size = int(config_args.config["model"]["batch_size"])
    dataset = FaceDataset(data_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    encoder = Encoder().to(device)
    inter = Inter().to(device)
    decoder_src = Decoder().to(device)
    decoder_dst = Decoder().to(device)

    optim_encoder = torch.optim.Adam([{"params": encoder.parameters()}, {"params": inter.parameters()}],
                                     lr=learning_rate)
    optim_decoder_src = torch.optim.Adam(decoder_src.parameters(), lr=learning_rate)
    optim_decoder_dst = torch.optim.Adam(decoder_dst.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    mean_loss_src = []
    mean_loss_dst = []

    if not new_model and (saved_models_dir_.absolute() / f'{model_name}.pth').exists():
        print(f"Loading Pretrained Model...: {saved_models_dir_.absolute() / f'{model_name}.pth'}")
        saved_model = torch.load(str(saved_models_dir_.absolute() / f'{model_name}.pth'))
        epoch = saved_model["epoch"]
    else:
        saved_model = {}
        epoch = 0
        mean_loss_src = []
        mean_loss_dst = []

    if saved_model:
        print("Loading Pretrained Model States...")
        encoder.load_state_dict(saved_model['encoder'])
        inter.load_state_dict(saved_model['inter'])
        decoder_src.load_state_dict(saved_model['decoder_src'])
        decoder_dst.load_state_dict(saved_model['decoder_dst'])
        optim_encoder.load_state_dict(saved_model['optimizer_encoder'])
        optim_decoder_src.load_state_dict(saved_model['optimizer_decoder_src'])
        optim_decoder_dst.load_state_dict(saved_model['optimizer_decoder_dst'])
        mean_loss_src = saved_model['mean_loss_src']
        mean_loss_dst = saved_model['mean_loss_dst']

    mean_epoch_loss_src = np.zeros(len(data_loader))
    mean_epoch_loss_dst = np.zeros(len(data_loader))

    encoder.train()
    inter.train()
    decoder_src.train()
    decoder_dst.train()

    first_run = True
    run = True

    print(f"Dataset Length per Data Loader: {len(data_loader.dataset)}")
    print(f"Data Loader Length: {len(data_loader)}")

    while run:
        epoch += 1
        for idx, (warp_im_src, target_im_src, warp_im_dst, target_im_dst) in enumerate(data_loader):
            # normalize
            warp_im_src /= 255.
            target_im_src /= 255.
            warp_im_dst /= 255.
            target_im_dst /= 255.
            # source image
            latent_sc = inter(encoder(warp_im_src))
            reconstruct_im_src = decoder_src(latent_sc)
            loss_dssim = sdsim(reconstruct_im_src, target_im_src)
            loss_mse = criterion(reconstruct_im_src, target_im_src)
            loss = loss_dssim + loss_mse
            optim_encoder.zero_grad()
            optim_decoder_src.zero_grad()

            loss.backward()
            optim_encoder.step()
            optim_decoder_src.step()
            loss_src = loss.item()

            # destination image
            latent_dst = inter(encoder(warp_im_dst))
            reconstruct_im_dst = decoder_dst(latent_dst)
            loss_dssim = sdsim(reconstruct_im_dst, target_im_dst)
            loss_mse = criterion(reconstruct_im_dst, target_im_dst)
            loss = loss_dssim + loss_mse
            optim_encoder.zero_grad()
            optim_decoder_dst.zero_grad()

            loss.backward()
            optim_encoder.step()
            optim_decoder_dst.step()
            loss_dst = loss.item()

            mean_epoch_loss_src[idx] = loss_src
            mean_epoch_loss_dst[idx] = loss_dst

            print(f"epoch: {epoch}, src_loss: {loss_src}, dst_loss: {loss_dst}")

            if first_run:
                first_run = False
                plt.ioff()
                fake = decoder_src(inter(encoder(target_im_dst)))

        if epoch % 100 == 0:
            saved_model['epoch'] = epoch
            saved_model['encoder'] = encoder.state_dict()
            saved_model['inter'] = inter.state_dict()
            saved_model['decoder_src'] = decoder_src.state_dict()
            saved_model['decoder_dst'] = decoder_dst.state_dict()
            saved_model['optimizer_encoder'] = optim_encoder.state_dict()
            saved_model['optimizer_decoder_src'] = optim_decoder_src.state_dict()
            saved_model['optimizer_decoder_dst'] = optim_decoder_dst.state_dict()
            saved_model['mean_loss_src'] = mean_loss_src
            saved_model['mean_loss_dst'] = mean_loss_dst
            saved_models_dir_.mkdir(exist_ok=True, parents=True)
            torch.save(saved_model, str(saved_models_dir_ / f'{model_name}.pth'))
            print(f"Model saved epoch: {epoch}")

        elif epoch % 10000 == 0:
            break

        mean_loss_src.append(mean_epoch_loss_src.mean())
        mean_loss_dst.append(mean_epoch_loss_dst.mean())
