import glob
import sys
import os
import losses
import utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from models import VxmDenseBPCA_2, VxmDenseBPCAMax_2, \
    VxmDenseMaxBPCA_2, VxmDenseBPCARevert_2, \
    VxmDenseMax_2, VxmDense_2
import random
import pystrum
import scipy


def main():
    test_dir = "/home/hinton/uemerson/OASIS/datasets/OASIS/Test/"

    models_dir = [
        "experiments/vxm_2_max_mse_1_diffusion_1/",
        "experiments/vxm_2_bpca_mse_1_diffusion_1/",
        "experiments/vxm_2_mse_1_diffusion_1/",
        "experiments/vxm_2_bpca_revert_mse_1_diffusion_1/",
        "experiments/vxm_2_max_bpca_mse_1_diffusion_1/",
        "experiments/vxm_2_bpca_max_mse_1_diffusion_1/",
    ]
    img_size = (160, 192, 224)
    models = [
        VxmDenseMax_2(img_size),
        VxmDenseBPCA_2(img_size),
        VxmDense_2(img_size),
        VxmDenseBPCARevert_2(img_size),
        VxmDenseMaxBPCA_2(img_size),
        VxmDenseBPCAMax_2(img_size),
    ]
    names = [
        "max",
        "bpca",
        "conv",
        "bpca-revert",
        "max-bpca",
        "bpca-max",
    ]
    #  ``'bilinear'`` | ``'nearest'`` | ``'bicubic'``
    reg_model_bilin = utils.register_model(img_size, "bilinear")
    reg_model_bilin.cuda()

    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType(
                                            (np.float32, np.int16)),
                                        ])
    test_set = datasets.OASISBrainInferDataset(
        glob.glob(test_dir + '*.pkl'), transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             num_workers=1, pin_memory=True, drop_last=True)
    can_break = False
    with torch.no_grad():
        for index_data, data in enumerate(test_loader):
            # 3
            # if index_data == 15:
            if index_data == 0:
                grid_index = 2
                # x, y, x_seg, y_seg = utils.pkload(data)

                # x, y = x[None, None, ...], y[None, None, ...]
                # x = np.ascontiguousarray(x)
                # y = np.ascontiguousarray(y)
                # x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
                # x_in = torch.cat((x, y), dim=1)
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                x_in = torch.cat((x, y), dim=1)

                grid = pystrum.pynd.ndutils.bw_grid(
                    vol_shape=img_size, spacing=8 if grid_index != 2 else 12, thickness=1)
                grid = grid[None, None, ...]

                results = []
                for index, model in enumerate(models):
                    print(f'Results of: {names[index]}')
                    print(models_dir[index])

                    model.cuda()
                    best_model = torch.load(models_dir[index] + natsorted(os.listdir(models_dir[index]))[-1])[
                        "state_dict"
                    ]
                    print("Best model: {}".format(
                        natsorted(os.listdir(models_dir[index]))[-1]))
                    model.load_state_dict(best_model)

                    model.eval()

                    x_def, flow = model(x_in)

                    def_grid = reg_model_bilin(
                        [torch.from_numpy(grid).cuda().float(), flow.cuda()])

                    slices_moving = [np.take(x.detach().cpu().numpy().squeeze(), img_size[d]//2, axis=d)
                                     for d in range(3)]
                    slices_moving[1] = np.rot90(slices_moving[1], 1)
                    slices_moving[2] = np.rot90(slices_moving[2], -1)

                    slices_fixed = [np.take(y.detach().cpu().numpy().squeeze(), img_size[d]//2, axis=d)
                                    for d in range(3)]
                    slices_fixed[1] = np.rot90(slices_fixed[1], 1)
                    slices_fixed[2] = np.rot90(slices_fixed[2], -1)

                    slices_pred = [np.take(x_def.detach().cpu().numpy().squeeze(), img_size[d]//2, axis=d)
                                   for d in range(3)]
                    slices_pred[1] = np.rot90(slices_pred[1], 1)
                    slices_pred[2] = np.rot90(slices_pred[2], -1)

                    slices_flow = [np.take(np.sum(flow.detach().cpu().numpy(), axis=1).squeeze(), img_size[d]//2, axis=d)
                                   for d in range(3)]
                    slices_flow[1] = np.rot90(slices_flow[1], 1)
                    slices_flow[2] = np.rot90(slices_flow[2], -1)

                    # def_grid = grid
                    # slices_grid = [np.take(def_grid.squeeze(), img_size[d]//2, axis=d)
                    #                for d in range(3)]

                    slices_grid = [np.take(def_grid.detach().cpu().numpy().squeeze(), img_size[d]//2, axis=d)
                                   for d in range(3)]
                    slices_grid[1] = np.rot90(slices_grid[1], 1)
                    slices_grid[2] = np.rot90(slices_grid[2], -1)

                    # flow_rgb = np.rot90(flow_as_rgb(flow, img_size[2]//2),  -1)
                    # flow_rgb = np.rot90(flow_as_rgb(flow, img_size[2]//2),  -1)

                    if grid_index == 0:
                        flow_rgb = flow_as_rgb(flow, img_size[2]//2)
                    elif grid_index == 1:
                        flow_rgb = np.rot90(
                            flow_as_rgb(flow, img_size[2]//2), 1)
                    else:
                        flow_rgb = np.rot90(flow_as_rgb(
                            flow, img_size[2]//2),  -1)

                    # grid_fig = comput_mult_fig(
                    #     [slices_moving[grid_index], slices_fixed[grid_index],
                    #      slices_pred[grid_index], slices_flow[grid_index], flow_rgb, slices_grid[grid_index]])
                    # grid_fig.savefig(
                    #     f'results-{names[index]}.png', bbox_inches='tight')

                    results.append([slices_moving[grid_index], slices_fixed[grid_index],
                                    slices_pred[grid_index], slices_flow[grid_index], flow_rgb, slices_grid[grid_index]])
                    del x_def, flow, def_grid
                    # flow_rgb = flow_as_rgb(slices_flow[2])
                    # flow_rgb = flow_as_rgb(flow)
                    # flow_a = gray_image_as_rgb(slices_flow[2])

                    # break

                if (grid_index == 0):
                    name_axis = "sagittal"
                elif (grid_index == 1):
                    name_axis = "axial"
                elif (grid_index == 2):
                    name_axis = "coronal"

                name_results = f"oasis_results_slice_{index_data}_{name_axis}.png"
                mult_results = comput_mult_figs(results)
                mult_results.savefig(name_results, bbox_inches='tight')
                can_break = True

            if can_break:
                break


def comput_mult_fig(imgs):
    fig, axs = plt.subplots(1, 6, figsize=(12, 12))
    for i in range(len(imgs)):
        axs[i].imshow(imgs[i], cmap='gray')
        axs[i].axis('off')
    fig.subplots_adjust(wspace=0.1, hspace=0)
    return fig


def comput_mult_figs(imgs):
    plt.rcParams['text.usetex'] = True

    fig, axs = plt.subplots(6, 6, figsize=(12, 12))
    # plt.rcParams['mathtext.fontset'] = 'cm'
    for i in range(6):
        for j in range(6):
            if i == 0:
                if j == 0:
                    axs[i, j].set_title('$m$', fontsize=20, pad=7)
                elif j == 1:
                    axs[i, j].set_title('$f$', fontsize=20, pad=7)
                elif j == 2:
                    axs[i, j].set_title(
                        r'$m \ {\scriptstyle \circ} \ \phi $', fontsize=20, pad=7, ha='center', va='center')
                else:
                    axs[i, j].set_title('$\phi$', fontsize=20, pad=7)

            # axs[i, j].imshow(imgs[i][j], cmap='gray', aspect="auto")
            axs[i, j].imshow(imgs[i][j], cmap='gray')
            axs[i, j].axis('off')

    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig


def flow_as_rgb(flow, slice_num):
    flow = flow.detach().cpu().numpy().squeeze()
    flow = flow[:, slice_num, :, :]
    flow_rgb = np.zeros((flow.shape[1], flow.shape[2], 3))
    for c in range(3):
        flow_rgb[..., c] = flow[c, :, :]
    lower = np.percentile(flow_rgb, 2)
    upper = np.percentile(flow_rgb, 98)
    flow_rgb[flow_rgb < lower] = lower
    flow_rgb[flow_rgb > upper] = upper
    flow_rgb = (((flow_rgb - flow_rgb.min()) /
                (flow_rgb.max() - flow_rgb.min())))

    # plt.figure()
    # plt.imshow(flow_rgb, vmin=0, vmax=1)
    # plt.axis('off')

    # plt.savefig('flow_rgb.png', bbox_inches='tight', pad_inches=0)
    # plt.show()

    return flow_rgb


def gray_image_as_rgb(gray_image):
    flow_rgb = np.zeros((gray_image.shape[0], gray_image.shape[1], 3))

    # Assuming the input gray_image is of shape (192, 160)
    for c in range(3):
        flow_rgb[..., c] = gray_image

    lower = np.percentile(flow_rgb, 2)
    upper = np.percentile(flow_rgb, 98)

    flow_rgb[flow_rgb < lower] = lower
    flow_rgb[flow_rgb > upper] = upper

    flow_rgb = (((flow_rgb - flow_rgb.min()) /
                 (flow_rgb.max() - flow_rgb.min())))

    # plt.figure()
    # plt.savefig('gray_image_as_rgb.png', bbox_inches='tight', pad_inches=0)

    # return flow_rgb


if __name__ == "__main__":
    """
    GPU configuration
    """
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print("Number of GPU: " + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print("     GPU #" + str(GPU_idx) + ": " + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print("Currently using: " + torch.cuda.get_device_name(GPU_iden))
    print("If the GPU is available? " + str(GPU_avai))

    # reproducibility
    seed = 42

    random.seed(seed)  # python random generator
    np.random.seed(seed)  # numpy random generator
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_iden)

    main()
