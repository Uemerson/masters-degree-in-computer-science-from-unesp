# Paper(s)

[Registro Não-Rígido de Imagens Médicas usando Block-Based Principal Component Analysis como Camada de Pooling](https://hdl.handle.net/11449/255307)

# About

This repository stores the training scripts, submission, and other files used in the research entitled `"Registro Não-Rígido de Imagens Médicas usando Block-Based Principal Component Analysis como Camada de Pooling"` conducted as part of the master's degree program at São Paulo State University (UNESP).

# Datasets

You can download the datasets used in this work from:

- [👉 Download IXI](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/IXI/TransMorph_on_IXI.md)
- [👉 Download OASIS](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration/blob/main/OASIS/TransMorph_on_OASIS.md)

# How to train ?

The train scripts are inside OASIS and IXI folder, start with train_*

## OASIS

The following scripts train using the OASIS dataset:

- `train_oasis_vxm_2.py`: Trains Voxelmorph without a pooling layer (convolution with stride).
- `train_oasis_vxm_2_max.py`: Trains Voxelmorph with a Max Pooling layer.
- `train_oasis_vxm_2_bpca.py`: Trains Voxelmorph with a BPCA (Block-Based Principal Component Analysis) pooling layer.
- `train_oasis_vxm_2_bpca_revert.py`: Trains Voxelmorph with a BPCA pooling layer and reverts BPCA on the unpooling layer.
- `train_oasis_vxm_2_max_bpca.py`: Trains Voxelmorph with a mix of the first 3 pooling layers using Max Pooling and the final layer using BPCA.
- `train_oasis_vxm_2_bpca_max.py`: Trains Voxelmorph with a mix of the first 3 pooling layers using BPCA and the final layer using Max Pooling.

## IXI

The following scripts train using the IXI dataset:

- `train_ixi_vxm_2.py`: Trains Voxelmorph without a pooling layer (convolution with stride).
- `train_ixi_vxm_2_max.py`: Trains Voxelmorph with a Max Pooling layer.
- `train_ixi_vxm_2_bpca.py`: Trains Voxelmorph with a BPCA (Block-Based Principal Component Analysis) pooling layer.
- `train_ixi_vxm_2_bpca_revert.py`: Trains Voxelmorph with a BPCA pooling layer and reverts BPCA on the unpooling layer.
- `train_ixi_vxm_2_max_bpca.py`: Trains Voxelmorph with a mix of the first 3 pooling layers using Max Pooling and the final layer using BPCA.
- `train_ixi_vxm_2_bpca_max.py`: Trains Voxelmorph with a mix of the first 3 pooling layers using BPCA and the final layer using Max Pooling.

# How to use tensorboard ?

After installing dependencies and activating environments using `conda`, you can:

```
$ tensorboard --logdir path-to-log/logs --port 6006
```

# Observation

In some scripts, you will need to change the foler path, for example, `/home/hinton/uemerson/` to your directory path.
