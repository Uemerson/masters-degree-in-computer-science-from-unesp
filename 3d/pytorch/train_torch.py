import sys
import os
import random
import argparse
import time
import numpy as np
import torch
import shutil
from torchsummary import summary


# reproducibility
seed = 7777

random.seed(seed)  # python random generator
np.random.seed(seed)  # numpy random generator
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def write(filename, text):
    try:
        # Extract the directory path from the filename
        directory = os.path.dirname(filename)

        # Check if the directory exists; if not, create it
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(filename):
            text = str(text)
        else:
            text = "\n" + str(text)

        with open(filename, "a") as file:
            file.write(text)

    except Exception as e:
        print(f"An error occurred: {str(e)}")


# parse the commandline
parser = argparse.ArgumentParser()

# data organization parameters
parser.add_argument(
    "--img-list", required=True, help="line-seperated list of training files"
)
parser.add_argument(
    "--val-list", required=True, help="line-seperated list of validating files"
)
parser.add_argument("--img-prefix", help="optional input image file prefix")
parser.add_argument("--img-suffix", help="optional input image file suffix")
parser.add_argument("--atlas", help="atlas filename (default: data/atlas_norm.npz)")
parser.add_argument(
    "--model-dir", default="models", help="model output directory (default: models)"
)
parser.add_argument(
    "--multichannel",
    action="store_true",
    help="specify that data has multiple channels",
)

# training parameters
parser.add_argument(
    "--gpu", default="0", help="GPU ID number(s), comma-separated (default: 0)"
)
parser.add_argument("--batch-size", type=int, default=1, help="batch size (default: 1)")
parser.add_argument(
    "--epochs", type=int, default=1500, help="number of training epochs (default: 1500)"
)
parser.add_argument(
    "--steps-per-epoch",
    type=int,
    default=100,
    help="frequency of model saves (default: 100)",
)
parser.add_argument("--load-model", help="optional model file to initialize with")
parser.add_argument(
    "--initial-epoch", type=int, default=0, help="initial epoch number (default: 0)"
)
parser.add_argument(
    "--lr", type=float, default=1e-4, help="learning rate (default: 1e-4)"
)
parser.add_argument(
    "--cudnn-nondet",
    action="store_true",
    help="disable cudnn determinism - might slow down training",
)

# network architecture parameters
parser.add_argument(
    "--enc",
    type=int,
    nargs="+",
    help="list of unet encoder filters (default: 16 32 32 32)",
)
parser.add_argument(
    "--dec",
    type=int,
    nargs="+",
    help="list of unet decorder filters (default: 32 32 32 32 32 16 16)",
)
parser.add_argument(
    "--int-steps", type=int, default=7, help="number of integration steps (default: 7)"
)
parser.add_argument(
    "--int-downsize",
    type=int,
    default=2,
    help="flow downsample factor for integration (default: 2)",
)
parser.add_argument(
    "--bidir", action="store_true", help="enable bidirectional cost function"
)

# loss hyperparameters
parser.add_argument(
    "--image-loss",
    default="mse",
    help="image reconstruction loss - can be mse or ncc (default: mse)",
)
parser.add_argument(
    "--lambda",
    type=float,
    dest="weight",
    default=0.01,
    help="weight of deformation loss (default: 0.01)",
)

# pooling methods


parser.add_argument(
    "--pooling-method",
    default="maxpooling",
    help="Pooling method: maxpooling, bpca or bpca-revert",
)

args = parser.parse_args()

if args.pooling_method == "maxpooling":
    shutil.copy(
        "/home/lecun/uemerson/voxelmorph/voxelmorph/torch/networks-maxpooling.py",
        "/home/lecun/uemerson/voxelmorph/voxelmorph/torch/networks.py",
    )

elif args.pooling_method == "bpca":
    shutil.copy(
        "/home/lecun/uemerson/voxelmorph/voxelmorph/torch/networks-bpca.py",
        "/home/lecun/uemerson/voxelmorph/voxelmorph/torch/networks.py",
    )

elif args.pooling_method == "bpca-revert":
    shutil.copy(
        "/home/lecun/uemerson/voxelmorph/voxelmorph/torch/networks-bpca-revert.py",
        "/home/lecun/uemerson/voxelmorph/voxelmorph/torch/networks.py",
    )

# import voxelmorph with pytorch backend
os.environ["NEURITE_BACKEND"] = "pytorch"
os.environ["VXM_BACKEND"] = "pytorch"
sys.path.append("voxelmorph")
import voxelmorph as vxm  # nopep8

bidir = args.bidir

# load and prepare training data
train_files = vxm.py.utils.read_file_list(
    args.img_list, prefix=args.img_prefix, suffix=args.img_suffix
)
val_files = vxm.py.utils.read_file_list(
    args.val_list, prefix=args.img_prefix, suffix=args.img_suffix
)
assert len(train_files) > 0, "Could not find any training data."
assert len(val_files) > 0, "Could not find any training data."

# no need to append an extra feature axis if data is multichannel
add_feat_axis = not args.multichannel

if args.atlas:
    # scan-to-atlas generator
    atlas = vxm.py.utils.load_volfile(
        args.atlas, np_var="vol", add_batch_axis=True, add_feat_axis=add_feat_axis
    )
    generator = vxm.generators.scan_to_atlas(
        train_files,
        atlas,
        batch_size=args.batch_size,
        bidir=args.bidir,
        add_feat_axis=add_feat_axis,
    )
    val_generator = vxm.generators.scan_to_atlas(
        val_files,
        atlas,
        batch_size=args.batch_size,
        bidir=args.bidir,
        add_feat_axis=add_feat_axis,
    )
else:
    # scan-to-scan generator
    generator = vxm.generators.scan_to_scan(
        train_files,
        batch_size=args.batch_size,
        bidir=args.bidir,
        add_feat_axis=add_feat_axis,
    )
    val_generator = vxm.generators.scan_to_scan(
        val_files,
        batch_size=args.batch_size,
        bidir=args.bidir,
        add_feat_axis=add_feat_axis,
    )

# extract shape from sampled input
inshape = next(generator)[0][0].shape[1:-1]

# prepare model folder
model_dir = args.model_dir
os.makedirs(model_dir, exist_ok=True)

# device handling
gpus = args.gpu.split(",")
nb_gpus = len(gpus)
device = "cuda"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
assert (
    np.mod(args.batch_size, nb_gpus) == 0
), "Batch size (%d) should be a multiple of the nr of gpus (%d)" % (
    args.batch_size,
    nb_gpus,
)

# enabling cudnn determinism appears to speed up training by a lot
torch.backends.cudnn.deterministic = not args.cudnn_nondet

# unet architecture
enc_nf = args.enc if args.enc else [16, 32, 32, 32]
dec_nf = args.dec if args.dec else [32, 32, 32, 32, 32, 16, 16]

if args.load_model:
    # load initial model (if specified)
    model = vxm.networks.VxmDense.load(args.load_model, device)
else:
    # otherwise configure new model
    model = vxm.networks.VxmDense(
        inshape=inshape,
        nb_unet_features=[enc_nf, dec_nf],
        bidir=bidir,
        int_steps=args.int_steps,
        int_downsize=args.int_downsize,
    )

if nb_gpus > 1:
    # use multiple GPUs via DataParallel
    model = torch.nn.DataParallel(model)
    model.save = model.module.save

if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is not available")


# prepare the model for training and send to device
model.to(device)
model.train()

# Summary model
summary(model, [(1, 160, 192, 224), (1, 160, 192, 224)])
# print(model)

# set optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# prepare image loss
if args.image_loss == "ncc":
    image_loss_func = vxm.losses.NCC().loss
elif args.image_loss == "mse":
    image_loss_func = vxm.losses.MSE().loss
else:
    raise ValueError(
        'Image loss should be "mse" or "ncc", but found "%s"' % args.image_loss
    )

# need two image loss functions if bidirectional
if bidir:
    losses = [image_loss_func, image_loss_func]
    weights = [0.5, 0.5]
else:
    losses = [image_loss_func]
    weights = [1]

# prepare deformation loss
losses += [vxm.losses.Grad("l2", loss_mult=args.int_downsize).loss]
weights += [args.weight]

print()
start_time = time.time()

num_val_samples = len(val_files)
batch_size = args.batch_size
validation_steps = int(np.ceil(num_val_samples / batch_size))

history_train_loss = []
history_val_loss = []

# training loops
for epoch in range(args.initial_epoch, args.epochs):
    step_start_epoch_time = time.time()

    model.train()  # Set the model in training mode

    epoch_loss = []
    epoch_total_loss = []
    epoch_val_loss = []
    epoch_total_val_loss = []
    epoch_step_time = []

    for step in range(args.steps_per_epoch):
        step_start_time = time.time()

        # generate inputs (and true outputs) and convert them to tensors
        inputs, y_true = next(generator)
        inputs = [
            torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3)
            for d in inputs
        ]
        y_true = [
            torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3)
            for d in y_true
        ]

        # run inputs through the model to produce a warped image and flow field
        y_pred = model(*inputs)

        # calculate total loss
        loss = 0
        loss_list = []
        for n, loss_function in enumerate(losses):
            curr_loss = loss_function(y_true[n], y_pred[n]) * weights[n]
            loss_list.append(curr_loss.item())
            loss += curr_loss

        epoch_loss.append(loss_list)
        epoch_total_loss.append(loss.item())

        # backpropagate and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # get compute time
        epoch_step_time.append(time.time() - step_start_time)

    model.eval()  # Set the model in evaluation mode

    for val_step in range(validation_steps):
        # Generate validation inputs (and true outputs) and convert them to tensors
        val_inputs, val_y_true = next(val_generator)
        val_inputs = [
            torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3)
            for d in val_inputs
        ]
        val_y_true = [
            torch.from_numpy(d).to(device).float().permute(0, 4, 1, 2, 3)
            for d in val_y_true
        ]
        # Forward pass for validation
        val_y_pred = model(*val_inputs)
        # calculate validation loss
        val_loss = 0
        val_loss_list = []
        for n, val_loss_function in enumerate(losses):
            curr_val_loss = loss_function(val_y_true[n], val_y_pred[n]) * weights[n]
            val_loss_list.append(curr_val_loss.item())
            val_loss += curr_val_loss

        epoch_val_loss.append(val_loss_list)
        epoch_total_val_loss.append(val_loss.item())

    # print epoch info
    epoch_info = "Epoch %d/%d" % (epoch + 1, args.epochs)
    time_info = "%.4f sec/step" % np.mean(epoch_step_time)
    losses_info = ", ".join(["%.4e" % f for f in np.mean(epoch_loss, axis=0)])
    loss_info = "loss: %.4e  (%s)" % (np.mean(epoch_total_loss), losses_info)
    val_losses_info = ", ".join(["%.4e" % f for f in np.mean(epoch_val_loss, axis=0)])
    val_loss_info = "val_loss: %.4e  (%s)" % (
        np.mean(epoch_total_val_loss),
        val_losses_info,
    )
    print(" - ".join((epoch_info, time_info, loss_info, val_loss_info)), flush=True)

    write("metrics/epoch_step_time.txt", np.mean(epoch_step_time))
    write("metrics/epoch_total_loss.txt", np.mean(epoch_total_loss))
    write("metrics/epoch_total_val_loss.txt", np.mean(epoch_total_val_loss))

    # save model checkpoint
    if epoch % 1 == 0:
        model.save(os.path.join(model_dir, "%04d.pt" % epoch))

    step_end_epoch_time = time.time()

    write(
        "metrics/step_end_epoch_time.txt", step_end_epoch_time - step_start_epoch_time
    )

end_time = time.time()

write("metrics/time_taken.txt", end_time - start_time)

print(f"Time taken: {end_time - start_time} seconds")

# final model save
model.save(os.path.join(model_dir, "%04d.pt" % args.epochs))
