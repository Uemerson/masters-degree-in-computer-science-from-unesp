import glob
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
from models import VxmDenseBPCAMax_2, VxmDense_2, VxmDenseMax_2, VxmDenseBPCA_2, VxmDenseBPCARevert_2, VxmDenseMaxBPCA_2
import torch.nn as nn
import random
import time


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)


def MSE_torch(x, y):
    return torch.mean((x - y) ** 2)


def MAE_torch(x, y):
    return torch.mean(torch.abs(x - y))


def main():
    atlas_dir = '/home/hinton/uemerson/IXI/datasets/IXI_data/atlas.pkl'
    test_dir = '/home/hinton/uemerson/IXI/datasets/IXI_data/Test/'
    model_idx = -1
    weights = [1, 1]
    model_folder = 'vxm_2_bpca_max_mse_{}_diffusion_{}/'.format(
        weights[0], weights[1])

    model_dir = 'experiments/' + model_folder
    if 'Val' in test_dir:
        csv_name = model_folder[:-1]+'_VAL'
    else:
        csv_name = model_folder[:-1]
    dict = utils.process_label()
    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/' + csv_name + '.csv'):
        os.remove('Quantitative_Results/' + csv_name + '.csv')
    csv_writter(model_folder[:-1], 'Quantitative_Results/' + csv_name)
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line + ',' + 'non_jec', 'Quantitative_Results/' + csv_name)
    model = VxmDenseBPCAMax_2((160, 192, 224))
    best_model = torch.load(
        model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model((160, 192, 224), 'bilinear')
    reg_model.cuda()
    test_composed = transforms.Compose([trans.Seg_norm(),
                                        trans.NumpyType(
                                            (np.float32, np.int16)),
                                        ])
    test_set = datasets.IXIBrainInferDataset(
        glob.glob(test_dir + '*.pkl'), atlas_dir, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()

    avg_time = 0
    total_interactions = len(test_loader)
    avg_jac = 0

    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            x_in = torch.cat((x, y), dim=1)
            start = time.time()
            x_def, flow = model(x_in)
            end = time.time()

            total_time = end - start
            avg_time = avg_time + total_time

            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            x_segs = []
            for i in range(46):
                def_seg = reg_model(
                    [x_seg_oh[:, i:i + 1, ...].float(), flow.float()])
                x_segs.append(def_seg)
            x_segs = torch.cat(x_segs, dim=1)
            def_out = torch.argmax(x_segs, dim=1, keepdim=True)
            del x_segs, x_seg_oh
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(
                flow.detach().cpu().numpy()[0, :, :, :, :])

            line = utils.dice_val_substruct(
                def_out.long(), y_seg.long(), stdy_idx)
            line = line + ',' + str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            csv_writter(line, 'Quantitative_Results/' + csv_name)
            eval_det.update(np.sum(jac_det <= 0) /
                            np.prod(tar.shape), x.size(0))

            avg_jac = avg_jac + np.sum(jac_det <= 0)

            dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}, Time: {:.4f}, Jac: {:.4f}'.format(dsc_trans.item(),
                                                                                         dsc_raw.item(), total_time,
                                                                                         np.sum(jac_det <= 0)))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1
        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {:.3f}, std: {:.3f}'.format(
            eval_det.avg, eval_det.std))
        print('avg time: {:.3f}, std time: {}'.format(
            avg_time/total_interactions))
        print('avg jac: {:.3f}, std jac:{}'.format(
            {avg_jac/total_interactions}))


def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')


if __name__ == '__main__':
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
