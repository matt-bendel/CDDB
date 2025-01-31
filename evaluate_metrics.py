import os
import argparse
from pathlib import Path
from easydict import EasyDict as edict

import numpy as np

import torch
import lpips
import os
import cv2

RESULT_DIR = Path("/storage/matt_models/ddb")


def compute_psnr(target, img2):
    target = np.clip(target, 0, 1)
    img2 = np.clip(img2, 0, 1)
    mse = np.mean((target - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-gpu-per-node", type=int, default=1, help="number of gpu on each node")
    parser.add_argument("--master-address", type=str, default='localhost', help="address for master")
    parser.add_argument("--method", type=str, default='cddb', help="name for method under test")
    parser.add_argument("--node-rank", type=int, default=0, help="the index of node")
    parser.add_argument("--num-proc-node", type=int, default=1, help="The number of nodes in multi node env")

    # data
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--dataset-dir", type=Path, default="/dataset", help="path to LMDB dataset")
    parser.add_argument("--partition", type=str, default=None, help="e.g., '0_4' means the first 25% of the dataset")
    parser.add_argument("--add-noise", action="store_true", help="If true, add small gaussian noise to y")

    # sample
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--ckpt", type=str, default=None, help="the checkpoint name from which we wish to sample")
    parser.add_argument("--nfe", type=int, default=None, help="sampling steps")
    parser.add_argument("--clip-denoise", action="store_true", help="clamp predicted image to [-1,1] at each")
    parser.add_argument("--use-fp16", action="store_true", help="use fp16 network weight for faster sampling")
    parser.add_argument("--eta", type=float, default=1.0, help="ddim stochasticity. 1.0 recovers ddpm")
    parser.add_argument("--use-cddb-deep", action="store_true", help="use cddb-deep")
    parser.add_argument("--use-cddb", action="store_true", help="use cddb")
    parser.add_argument("--no-reg", action="store_true", help="use original I2SB weights")
    parser.add_argument("--step-size", type=float, default=1.0, help="step size for gradient descent")
    parser.add_argument("--prob_mask", type=float, default=0.35, help="probability of masking")

    arg = parser.parse_args()

    opt = edict(
        distributed=(arg.n_gpu_per_node > 1),
        device="cuda",
    )
    opt.update(vars(arg))

    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()

    PSNR_all_imgs = torch.zeros(1000, 1)
    LPIPS_all_imgs = torch.zeros(1000, 1)

    total_num_of_images = 100

    gt_test_image_folder = f"/storage/matt_models/ddb/{opt.method}/{opt.ckpt}/samples/label/"
    recon_image_folder = f"/storage/matt_models/ddb/{opt.method}/{opt.ckpt}/samples/recon/"

    # TODO: SSIM
    for test_image_number in range(total_num_of_images):
        num_str = f"{test_image_number:03}.png"

        gt_img_path = os.path.join(gt_test_image_folder, num_str)
        output_img_path = os.path.join(recon_image_folder, num_str)

        # Read images
        gt_image_0_255 = cv2.imread(gt_img_path, cv2.IMREAD_COLOR)
        recon_img_0_255 = cv2.imread(output_img_path, cv2.IMREAD_COLOR)

        gt_image = gt_image_0_255 / 255
        recon_img = recon_img_0_255 / 255

        # if test_image_number%50 == 0:
        #     print(f"Processing image {test_image_number}/1000")
        psnr = compute_psnr(gt_image, recon_img)
        print(psnr)
        PSNR_all_imgs[test_image_number, 0] = psnr

        x_m1t1 = torch.from_numpy((2 * (1 * gt_image_0_255 / 255) - 1)).permute(2, 0, 1).unsqueeze(
            0).contiguous().float().cuda()
        sample_m1t1 = torch.from_numpy(2 * (1 * recon_img_0_255 / 255) - 1).permute(2, 0, 1).unsqueeze(
            0).contiguous().float().cuda()

        LPIPS_all_imgs[test_image_number, 0] = loss_fn_vgg(sample_m1t1, x_m1t1).mean().detach().cpu()

    # Average PSNR and LPIPS
    print("=====================================")
    PSNR_avg_np_round = np.round(PSNR_all_imgs[:-1].mean().numpy(), 2)
    LPIPS_avg_np_round = np.round(LPIPS_all_imgs[:-1].mean().numpy(), 4)
    # print(" ")
    # PSNR_all_imgs_mean = np.round(torch.mean(PSNR_all_imgs[:,0]).numpy(),2)
    print("Average PSNR  :", PSNR_avg_np_round)
    print("Average LPIPS :", LPIPS_avg_np_round)
    # print(" ")
    print("=====================================")
