import cv2
import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import gridspec
import matplotlib.patches as patches

ncol = 10
num_rows_per_fig = 3
num_figs_per_prob = 3

probs = ['sr4x-bicubic', 'inpaint-center']
nfes = [20, 1000]
methods = ['baseline-128', 'unrolled-128', 'hybrid-128']


def load_images_as_np_array(path, offset, num_ims, im_name):
    ims = []
    for idx in range(num_ims):
        adjusted_idx = offset * num_ims + idx
        im_path = path + '/' + im_name(adjusted_idx)

        img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)  # BGR or G
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

        ims.append(img)

    return ims


def load_image_as_np_array(im_path):
    img = cv2.imread(im_path, cv2.IMREAD_UNCHANGED)  # BGR or G
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB

    return img


im_path = lambda nfe_str, prob, method: f'/storage/matt_models/ddb/{method}/{prob}-{nfe_str}/samples/recon'
save_path = 'figures'
for problem in probs:
    im_idx = 0
    fig_count = 0

    for i in range(num_figs_per_prob):
        fig = plt.figure(figsize=(ncol, num_rows_per_fig + 1))

        gs = gridspec.GridSpec(num_rows_per_fig + 1, ncol,
                               wspace=0.0, hspace=0.0)

        for j in range(num_rows_per_fig):
            new_num = 10 * im_idx + 9
            if new_num < 1000:
                num_str = f"{new_num:03}.png"
            else:
                num_str = f"{new_num}.png"

            # GT Image
            ax = plt.subplot(gs[j, 0])
            im = ax.imshow(load_image_as_np_array(f'/storage/matt_models/ddb/baseline-128/sr4x-bicubic-1000/samples/label/{num_str}'))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

            # Measurements
            ax = plt.subplot(gs[j, 1])
            im = ax.imshow(load_image_as_np_array(
                f'/storage/matt_models/ddb/baseline-128/{problem}-1000/samples/input/{num_str}'))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

            col_index = 2
            for nfe_str in nfes:
                col_index += 1

                for method in methods:
                    # Measurements
                    ax = plt.subplot(gs[j, col_index])
                    im = ax.imshow(load_image_as_np_array(
                        f'{im_path(nfe_str, problem, method)}/{num_str}'))
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_xticks([])
                    ax.set_yticks([])
                    col_index += 1

            im_idx += 1

        plt.savefig(f'figures/{problem}_fig_{fig_count}.png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        fig_count += 1

