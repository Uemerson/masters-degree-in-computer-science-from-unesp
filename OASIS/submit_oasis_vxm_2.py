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
from models import VxmDense_2
import random


def main():
    # test_dir = "/home/hinton/uemerson/datasets/OASIS/Test/"
    test_dir = "/home/hinton/uemerson/OASIS/datasets/OASIS/Challenge_test_no_gt/"
    save_dir = "/home/hinton/uemerson/OASIS/submit/vxm_2_mse_1_diffusion_1/task_03/"
    model_idx = -1
    weights = [1, 1]
    model_folder = "vxm_2_mse_{}_diffusion_{}/".format(
        weights[0], weights[1])
    model_dir = "experiments/" + model_folder
    img_size = (160, 192, 224)
    model = VxmDense_2(img_size)
    model.cuda()
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])[
        "state_dict"
    ]
    print("Best model: {}".format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(img_size, "nearest")
    reg_model.cuda()
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
            # x, y, x_seg, y_seg = utils.pkload(data)
            x, y = utils.pkload(data)
            x, y = x[None, None, ...], y[None, None, ...]
            x = np.ascontiguousarray(x)
            y = np.ascontiguousarray(y)
            x, y = torch.from_numpy(x).cuda(), torch.from_numpy(y).cuda()
            file_name = os.path.basename(data).replace(
                "p_", "").replace(".pkl", "")
            model.eval()
            x_in = torch.cat((x, y), dim=1)
            x_def, flow = model(x_in)
            flow = flow.cpu().detach().numpy()[0]
            flow = np.array([zoom(flow[i], 0.5, order=2) for i in range(3)]).astype(
                np.float16
            )
            np.savez(save_dir + "disp_{}.npz".format(file_name), flow)


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
