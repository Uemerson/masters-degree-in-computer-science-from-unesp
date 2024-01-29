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
from scipy.ndimage.interpolation import map_coordinates, zoom
from models import VxmDenseBPCA_2
import random
import pystrum
import scipy


def main():
    test_dir = "/home/hinton/uemerson/OASIS/datasets/OASIS/Test/"
    # test_dir = "/home/hinton/uemerson/datasets/OASIS/Challenge_test_no_gt/"
    # save_dir = "/home/hinton/uemerson/OASIS/register/vxm_2_bpca_mse_1_diffusion_1/task_03/"
    model_idx = -1
    weights = [1, 1]
    model_folder = "vxm_2_bpca_mse_{}_diffusion_{}/".format(
        weights[0], weights[1])
    model_dir = "experiments/" + model_folder
    img_size = (160, 192, 224)

    model = VxmDenseBPCA_2(img_size)
    model.cuda()
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])[
        "state_dict"
    ]
    print("Best model: {}".format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(img_size, "nearest")
    reg_model.cuda()

    reg_model_bilin = utils.register_model(img_size, "bilinear")
    reg_model_bilin.cuda()

    test_composed = transforms.Compose(
        [
            trans.NumpyType((np.float32, np.int16)),
        ]
    )
    test_set = datasets.OASISBrainInferDataset(
        glob.glob(test_dir + "*.pkl"), transforms=test_composed
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=True,
    )
    file_names = glob.glob(test_dir + "*.pkl")
    with torch.no_grad():
        for data in file_names:
            x, y, x_seg, y_seg = utils.pkload(data)

            # x, y = utils.pkload(data)
            x, y = x[None, None, ...], y[None, None, ...]
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()

            model.eval()
            x_in = torch.cat((x, y), dim=1)
            x_def, flow = model(x_in)

            # grid_img = mk_grid_img(8, 1, img_size)
            # grid = pystrum.pynd.ndutils.bw_grid(vol_shape=img_size, spacing=11)
            grid = pystrum.pynd.ndutils.bw_grid(
                vol_shape=img_size, spacing=11, thickness=2)
            grid = grid[None, None, ...]
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

            slices_grid = [np.take(def_grid.detach().cpu().numpy().squeeze(), img_size[d]//2, axis=d)
                           for d in range(3)]
            slices_grid[1] = np.rot90(slices_grid[1], 1)
            slices_grid[2] = np.rot90(slices_grid[2], -1)

            slices_grid = [np.take(def_grid.detach().cpu().numpy().squeeze(), img_size[d]//2, axis=d)
                           for d in range(3)]
            slices_grid[1] = np.rot90(slices_grid[1], 1)
            slices_grid[2] = np.rot90(slices_grid[2], -1)

            grid_fig = comput_mult_fig(
                [slices_moving[2], slices_fixed[2], slices_pred[2], slices_grid[2]])
            grid_fig.savefig('grid.png')

            break


def comput_mult_fig(imgs):
    fig = plt.figure(figsize=(12, 12))

    for i in range(len(imgs)):
        plt.subplot(len(imgs), 4, i + 1)
        plt.axis("off")
        plt.imshow(imgs[i], cmap="gray")

    # plt.axis("off")
    # plt.imshow(img, cmap="gray")
    fig.subplots_adjust(wspace=0.05, hspace=0)
    return fig


def comput_fig(img):
    fig = plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.imshow(img, cmap="gray")
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def mk_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j + line_thickness - 1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def flow_as_rgb(flow, slice_num):
    # 1, 3, 160, 192, 224
    flow = flow[:, :, slice_num, :, :]
    flow_rgb = np.zeros((flow.shape[1], flow.shape[2], 3))
    for c in range(3):
        flow_rgb[..., c] = flow[c, :, :]
    lower = np.percentile(flow_rgb, 2)
    upper = np.percentile(flow_rgb, 98)
    flow_rgb[flow_rgb < lower] = lower
    flow_rgb[flow_rgb > upper] = upper
    flow_rgb = (((flow_rgb - flow_rgb.min()) /
                (flow_rgb.max() - flow_rgb.min())))
    plt.figure()
    plt.imshow(flow_rgb, vmin=0, vmax=1)
    plt.axis('off')
    plt.show()
    print(lower)
    print(upper)


# def comput_fig(img):
#     # img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
#     img = img.detach().cpu().numpy()[0, 0, 160//2, :, :]
#     # img = img.detach().cpu().numpy()[0, 0, 160//2:(160//2)+1, :, :]
#     # fig = plt.figure(figsize=(12, 12), dpi=180)
#     fig = plt.figure(figsize=(12, 12))
#     # print(img.shape, 'img')
#     # for i in range(img.shape[0]):
#     #     plt.subplot(4, 4, i + 1)
#     #     plt.axis("off")
#     #     plt.imshow(img[i, :, :], cmap="gray")

#     plt.axis("off")
#     plt.imshow(img, cmap="gray")
#     fig.subplots_adjust(wspace=0, hspace=0)
#     return fig


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
