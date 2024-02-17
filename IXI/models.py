"""
VoxelMorph

Original code retrieved from:
https://github.com/voxelmorph/voxelmorph

Original paper:
Balakrishnan, G., Zhao, A., Sabuncu, M. R., Guttag, J., & Dalca, A. V. (2019).
VoxelMorph: a learning framework for deformable medical image registration.
IEEE transactions on medical imaging, 38(8), 1788-1800.

Modified and tested by:
Junyu Chen
jchen245@jhmi.edu
Johns Hopkins University
"""

import torch
import torch.nn as nn
import torch.nn.functional as nnf

import numpy as np
from torch.distributions.normal import Normal


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * \
                (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class BPCA3D(nn.Module):
    """Custom BPCA 3D"""

    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1, n_components=1):
        super(BPCA3D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.n_components = n_components

    def extract_patches_3d(self, x, kernel_size, padding=0, stride=1, dilation=1):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        def get_dim_blocks(
            dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1
        ):
            dim_out = (
                dim_in + 2 * dim_padding - dim_dilation *
                (dim_kernel_size - 1) - 1
            ) // dim_stride + 1
            return dim_out

        channels = x.shape[1]

        d_dim_in = x.shape[2]
        h_dim_in = x.shape[3]
        w_dim_in = x.shape[4]
        d_dim_out = get_dim_blocks(
            d_dim_in, kernel_size[0], padding[0], stride[0], dilation[0]
        )
        h_dim_out = get_dim_blocks(
            h_dim_in, kernel_size[1], padding[1], stride[1], dilation[1]
        )
        w_dim_out = get_dim_blocks(
            w_dim_in, kernel_size[2], padding[2], stride[2], dilation[2]
        )
        # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

        # (B, C, D, H, W)
        x = x.view(-1, channels, d_dim_in, h_dim_in * w_dim_in)
        # (B, C, D, H * W)

        x = torch.nn.functional.unfold(
            x,
            kernel_size=(kernel_size[0], 1),
            padding=(padding[0], 0),
            stride=(stride[0], 1),
            dilation=(dilation[0], 1),
        )
        # (B, C * kernel_size[0], d_dim_out * H * W)

        x = x.view(-1, channels * kernel_size[0]
                   * d_dim_out, h_dim_in, w_dim_in)
        # (B, C * kernel_size[0] * d_dim_out, H, W)

        x = torch.nn.functional.unfold(
            x,
            kernel_size=(kernel_size[1], kernel_size[2]),
            padding=(padding[1], padding[2]),
            stride=(stride[1], stride[2]),
            dilation=(dilation[1], dilation[2]),
        )
        # (B, C * kernel_size[0] * d_dim_out * kernel_size[1] * kernel_size[2], h_dim_out, w_dim_out)

        x = x.view(
            -1,
            channels,
            kernel_size[0],
            d_dim_out,
            kernel_size[1],
            kernel_size[2],
            h_dim_out,
            w_dim_out,
        )
        # (B, C, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)

        x = x.permute(0, 1, 3, 6, 7, 2, 4, 5)
        # (B, C, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])

        x = x.contiguous().view(
            -1, channels, kernel_size[0], kernel_size[1], kernel_size[2]
        )
        # (B * d_dim_out * h_dim_out * w_dim_out, C, kernel_size[0], kernel_size[1], kernel_size[2])

        return x

    def pca(self, data):
        n_components = self.n_components

        # Normalize the data by subtracting the mean
        mean = torch.mean(data, dim=0)
        data = data - mean

        # Perform the Singular Value Decomposition (SVD) on the data
        _, _, v = torch.svd(data)

        # Extract the first n principal components from the matrix v
        pca_components = v[:, :n_components]

        # Perform the PCA transformation on the data
        reduced_data = torch.matmul(data, pca_components)

        return reduced_data, mean, pca_components

    def forward(self, x):
        """
        b: batch
        c: channels
        s: slices
        h: height
        w: width
        """
        b, c, s, w, h = x.shape
        blocks = self.extract_patches_3d(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
            dilation=self.dilation,
        )
        d = (b * c) // (self.kernel_size * self.kernel_size * self.kernel_size)
        data = blocks.reshape(
            h * w * s * d, (self.kernel_size *
                            self.kernel_size * self.kernel_size)
        )
        pca, mean, pca_components = self.pca(data)
        return (
            pca.reshape(
                b,
                c,
                s // self.kernel_size,
                w // self.kernel_size,
                h // self.kernel_size,
            ),
            mean,
            pca_components,
        )


class UpsampleBPCA3D(nn.Module):
    """Custom Upsample BPCA 3D"""

    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1, n_components=1):
        super(UpsampleBPCA3D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.n_components = n_components

    def combine_patches_3d(
        self, x, kernel_size, output_shape, padding=0, stride=1, dilation=1
    ):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        def get_dim_blocks(
            dim_in, dim_kernel_size, dim_padding=0, dim_stride=1, dim_dilation=1
        ):
            dim_out = (
                dim_in + 2 * dim_padding - dim_dilation *
                (dim_kernel_size - 1) - 1
            ) // dim_stride + 1
            return dim_out

        channels = x.shape[1]
        d_dim_out, h_dim_out, w_dim_out = output_shape[2:]
        d_dim_in = get_dim_blocks(
            d_dim_out, kernel_size[0], padding[0], stride[0], dilation[0]
        )
        h_dim_in = get_dim_blocks(
            h_dim_out, kernel_size[1], padding[1], stride[1], dilation[1]
        )
        w_dim_in = get_dim_blocks(
            w_dim_out, kernel_size[2], padding[2], stride[2], dilation[2]
        )
        # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

        x = x.view(
            -1,
            channels,
            d_dim_in,
            h_dim_in,
            w_dim_in,
            kernel_size[0],
            kernel_size[1],
            kernel_size[2],
        )
        # (B, C, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])

        x = x.permute(0, 1, 5, 2, 6, 7, 3, 4)
        # (B, C, kernel_size[0], d_dim_in, kernel_size[1], kernel_size[2], h_dim_in, w_dim_in)

        x = x.contiguous().view(
            -1,
            channels * kernel_size[0] * d_dim_in *
            kernel_size[1] * kernel_size[2],
            h_dim_in * w_dim_in,
        )
        # (B, C * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)

        x = torch.nn.functional.fold(
            x,
            output_size=(h_dim_out, w_dim_out),
            kernel_size=(kernel_size[1], kernel_size[2]),
            padding=(padding[1], padding[2]),
            stride=(stride[1], stride[2]),
            dilation=(dilation[1], dilation[2]),
        )
        # (B, C * kernel_size[0] * d_dim_in, H, W)

        x = x.view(-1, channels *
                   kernel_size[0], d_dim_in * h_dim_out * w_dim_out)
        # (B, C * kernel_size[0], d_dim_in * H * W)

        x = torch.nn.functional.fold(
            x,
            output_size=(d_dim_out, h_dim_out * w_dim_out),
            kernel_size=(kernel_size[0], 1),
            padding=(padding[0], 0),
            stride=(stride[0], 1),
            dilation=(dilation[0], 1),
        )
        # (B, C, D, H * W)

        x = x.view(-1, channels, d_dim_out, h_dim_out, w_dim_out)
        # (B, C, D, H, W)

        return x

    def forward(self, x, output_shape, mean, pca_components):
        # pca_shape,
        b, c, s, w, h = x.shape
        data = x.reshape(b * c * s * w * h, self.n_components)

        # Revert PCA transformation
        reverted_patches = torch.matmul(data, pca_components.t())

        # Add back the mean
        reverted_patches = reverted_patches + mean

        reverted_patches = reverted_patches.reshape(
            b, c, s * self.kernel_size, w * self.kernel_size, h * self.kernel_size
        )

        combine_data = self.combine_patches_3d(
            x=reverted_patches,
            kernel_size=self.kernel_size,
            output_shape=output_shape,
            padding=self.padding,
            stride=self.stride,
        )

        return combine_data


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(self, ndims, in_channels, out_channels, stride=1):
        super().__init__()

        Conv = getattr(nn, "Conv%dd" % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.main(x)
        out = self.activation(out)
        return out


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:
        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self, inshape, nb_features=None, nb_levels=None, feat_mult=1):
        super().__init__()
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
        """

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = ((8, 32, 32, 32), (32, 32, 32, 32, 32, 8, 8))

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError(
                    "must provide unet nb_levels if nb_features is an integer"
                )
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(
                int
            )
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError(
                "cannot use nb_levels if nb_features is not an integer")
        else:
            self.enc_nf, self.dec_nf = nb_features

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # configure encoder (down-sampling path)
        prev_nf = 2
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(ndims, prev_nf, nf, stride=2))
            prev_nf = nf

        # configure decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        self.uparm = nn.ModuleList()
        for i, nf in enumerate(self.dec_nf[: len(self.enc_nf)]):
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1))
            prev_nf = nf

        # configure extra decoder convolutions (no up-sampling)
        prev_nf += 2
        self.extras = nn.ModuleList()
        for nf in self.dec_nf[len(self.enc_nf):]:
            self.extras.append(ConvBlock(ndims, prev_nf, nf, stride=1))
            prev_nf = nf

    def forward(self, x):
        # get encoder activations
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        # conv, upsample, concatenate series
        x = x_enc.pop()
        for layer in self.uparm:
            x = layer(x)
            x = self.upsample(x)
            x = torch.cat([x, x_enc.pop()], dim=1)

        # extra convs at full resolution
        for layer in self.extras:
            x = layer(x)

        return x


class UnetBPCA(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(
        self,
        inshape=None,
        nb_features=None,
        nb_levels=None,
        feat_mult=1,
        half_res=False,
        nb_conv_per_level=1,
        max_pool=2,
        infeats=None,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        print("U-net BPCA with nearest upsample")

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = [
                [16, 32, 32, 32],             # encoder
                [32, 32, 32, 32, 32, 16, 16]  # decoder
            ]

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError(
                    "must provide unet nb_levels if nb_features is an integer"
                )
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(
                int
            )
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level),
            ]
        elif nb_levels is not None:
            raise ValueError(
                "cannot use nb_levels if nb_features is not an integer")

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        bpca3d = BPCA3D()
        self.pooling = [bpca3d for _ in max_pool]
        self.upsampling = [
            nn.Upsample(scale_factor=s, mode="nearest") for s in max_pool
        ]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        mean_history = []
        pca_components_history = []
        output_shape_history = []
        total_conv = 0
        total_pooling = 0
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
                total_conv = total_conv + 1
            x_history.append(x)
            output_shape_history.append(x.shape)
            x, mean, pca_components = self.pooling[level](x)
            mean_history.append(mean)
            pca_components_history.append(pca_components)
            total_pooling = total_pooling + 1

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class UnetMax(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(
        self,
        inshape=None,
        nb_features=None,
        nb_levels=None,
        feat_mult=1,
        half_res=False,
        nb_conv_per_level=1,
        max_pool=2,
        infeats=None,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        print("U-net MaxPooling with nearest upsample")

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = [
                [16, 32, 32, 32],             # encoder
                [32, 32, 32, 32, 32, 16, 16]  # decoder
            ]

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError(
                    "must provide unet nb_levels if nb_features is an integer"
                )
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(
                int
            )
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level),
            ]
        elif nb_levels is not None:
            raise ValueError(
                "cannot use nb_levels if nb_features is not an integer")

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(
            scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class UnetMaxBPCA(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(
        self,
        inshape=None,
        nb_features=None,
        nb_levels=None,
        feat_mult=1,
        half_res=False,
        nb_conv_per_level=1,
        max_pool=2,
        infeats=None,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        print("U-net BPCA with nearest upsample")

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = [
                [16, 32, 32, 32],             # encoder
                [32, 32, 32, 32, 32, 16, 16]  # decoder
            ]

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError(
                    "must provide unet nb_levels if nb_features is an integer"
                )
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(
                int
            )
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level),
            ]
        elif nb_levels is not None:
            raise ValueError(
                "cannot use nb_levels if nb_features is not an integer")

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        bpca3d = BPCA3D()
        self.pooling = [MaxPooling(s)
                        for s in max_pool][:-2] + [bpca3d, bpca3d]
        self.upsampling = [nn.Upsample(
            scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        mean_history = []
        pca_components_history = []
        output_shape_history = []
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            if isinstance(self.pooling[level], nn.MaxPool3d):
                x_history.append(x)
                x = self.pooling[level](x)
            elif isinstance(self.pooling[level], BPCA3D):
                x_history.append(x)
                output_shape_history.append(x.shape)
                x, mean, pca_components = self.pooling[level](x)
                mean_history.append(mean)
                pca_components_history.append(pca_components)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class UnetBPCAMax(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(
        self,
        inshape=None,
        nb_features=None,
        nb_levels=None,
        feat_mult=1,
        half_res=False,
        nb_conv_per_level=1,
        max_pool=2,
        infeats=None,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        print("U-net BPCA (3) + Max (2) with nearest upsample")

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = [
                [16, 32, 32, 32],             # encoder
                [32, 32, 32, 32, 32, 16, 16]  # decoder
            ]

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError(
                    "must provide unet nb_levels if nb_features is an integer"
                )
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(
                int
            )
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level),
            ]
        elif nb_levels is not None:
            raise ValueError(
                "cannot use nb_levels if nb_features is not an integer")

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        bpca3d = BPCA3D()
        self.pooling = [bpca3d, bpca3d, bpca3d] + \
            [MaxPooling(s) for s in max_pool][:-3]
        self.upsampling = [nn.Upsample(
            scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        mean_history = []
        pca_components_history = []
        output_shape_history = []
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            if isinstance(self.pooling[level], nn.MaxPool3d):
                x_history.append(x)
                x = self.pooling[level](x)
            elif isinstance(self.pooling[level], BPCA3D):
                x_history.append(x)
                output_shape_history.append(x.shape)
                x, mean, pca_components = self.pooling[level](x)
                mean_history.append(mean)
                pca_components_history.append(pca_components)
        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class UnetBPCARevert(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(
        self,
        inshape=None,
        infeats=None,
        nb_features=None,
        nb_levels=None,
        max_pool=2,
        feat_mult=1,
        nb_conv_per_level=1,
        half_res=False,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer.
                If None (default), the unet features are defined by the default config described in
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer.
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer.
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        print("U-net BPCA with revert BPCA upsampling")

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = [
                [16, 32, 32, 32],  # encoder
                [32, 32, 32, 32, 32, 16, 16],  # decoder
            ]

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError(
                    "must provide unet nb_levels if nb_features is an integer"
                )
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(
                int
            )
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level),
            ]
        elif nb_levels is not None:
            raise ValueError(
                "cannot use nb_levels if nb_features is not an integer")

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        bpca3d = BPCA3D()
        upsample_bpca3d = UpsampleBPCA3D()
        self.pooling = [bpca3d for _ in max_pool]
        self.upsampling = [upsample_bpca3d for _ in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(ndims, prev_nf, nf))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(ndims, prev_nf, nf))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

    def forward(self, x):
        # encoder forward pass
        x_history = [x]
        x_history_pca = []
        mean_history = []
        pca_components_history = []
        output_shape_history = []
        for level, convs in enumerate(self.encoder):
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            output_shape_history.append(x.shape)
            # x = self.pooling[level](x)
            x, mean, pca_components = self.pooling[level](x)
            x_history_pca.append(x)
            mean_history.append(mean)
            pca_components_history.append(pca_components)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                # x = self.upsampling[level](x)
                # x = torch.cat([x, x_history.pop()], dim=1)

                index = len(mean_history) - 1 - level
                x = self.upsampling[level](
                    x_history_pca[index],
                    output_shape_history[index],
                    mean_history[index],
                    pca_components_history[index],
                )
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class VxmDense_1(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(
        self,
        inshape,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.flow = Conv(
            self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                "Flow variance has not been implemented in pytorch - set use_probs to False"
            )

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        """

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow


class VxmDense_2(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(
        self,
        inshape,
        # nb_unet_features=((16, 32, 32, 32), (32, 32, 32, 32, 32, 16, 16)),
        nb_unet_features=((32, 32, 32, 32), (32, 32, 32, 32, 32, 16, 16)),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.flow = Conv(
            self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                "Flow variance has not been implemented in pytorch - set use_probs to False"
            )

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        """

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow


class VxmDenseMax_2(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(
        self,
        inshape,
        nb_unet_features=((32, 32, 32, 32), (32, 32, 32, 32, 32, 16, 16)),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # configure core unet model
        self.unet_model = UnetMax(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            infeats=2
        )

        # configure unet to flow field layer
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.flow = Conv(self.unet_model.final_nf,
                         ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                "Flow variance has not been implemented in pytorch - set use_probs to False"
            )

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        """

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow


class VxmDenseBPCA_2(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(
        self,
        inshape,
        # nb_unet_features=((16, 32, 32, 32), (32, 32, 32, 32, 32, 16, 16)),
        nb_unet_features=((32, 32, 32, 32), (32, 32, 32, 32, 32, 16, 16)),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # configure core unet model
        self.unet_model = UnetBPCA(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            infeats=2
        )

        # configure unet to flow field layer
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.flow = Conv(self.unet_model.final_nf,
                         ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                "Flow variance has not been implemented in pytorch - set use_probs to False"
            )

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        """

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow


class VxmDenseMaxBPCA_2(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(
        self,
        inshape,
        nb_unet_features=((32, 32, 32, 32), (32, 32, 32, 32, 32, 16, 16)),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # configure core unet model
        self.unet_model = UnetMaxBPCA(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            infeats=2
        )

        # configure unet to flow field layer
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.flow = Conv(self.unet_model.final_nf,
                         ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                "Flow variance has not been implemented in pytorch - set use_probs to False"
            )

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        """

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow


class VxmDenseBPCAMax_2(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(
        self,
        inshape,
        nb_unet_features=((32, 32, 32, 32), (32, 32, 32, 32, 32, 16, 16)),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # configure core unet model
        self.unet_model = UnetBPCAMax(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            infeats=2
        )

        # configure unet to flow field layer
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.flow = Conv(self.unet_model.final_nf,
                         ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                "Flow variance has not been implemented in pytorch - set use_probs to False"
            )

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        """

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow


class VxmDenseBPCARevert_2(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(
        self,
        inshape,
        nb_unet_features=((32, 32, 32, 32), (32, 32, 32, 32, 32, 16, 16)),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # configure core unet model
        self.unet_model = UnetBPCARevert(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            infeats=2,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.flow = Conv(self.unet_model.final_nf,
                         ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                "Flow variance has not been implemented in pytorch - set use_probs to False"
            )

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        """

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow


class VxmDensex2(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(
        self,
        inshape,
        nb_unet_features=((32, 64, 64, 64), (64, 64, 64, 64, 64, 32, 32)),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.flow = Conv(
            self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                "Flow variance has not been implemented in pytorch - set use_probs to False"
            )

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        """

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow


class VxmDense_huge(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """

    def __init__(
        self,
        inshape,
        nb_unet_features=((14, 28, 144, 320),
                          (1152, 1152, 320, 144, 28, 14, 14)),
        nb_unet_levels=None,
        unet_feat_mult=1,
        use_probs=False,
    ):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. If None (default),
                the unet features are defined by the default config described in the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. The flow field
                is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.
            use_probs: Use probabilities in flow field. Default is False.
        """
        super().__init__()

        # internal flag indicating whether to return flow or integrated warp during inference
        self.training = True

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], (
            "ndims should be one of 1, 2, or 3. found: %d" % ndims
        )

        # configure core unet model
        self.unet_model = Unet(
            inshape,
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
        )

        # configure unet to flow field layer
        Conv = getattr(nn, "Conv%dd" % ndims)
        self.flow = Conv(
            self.unet_model.dec_nf[-1], ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(
            Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # probabilities are not supported in pytorch
        if use_probs:
            raise NotImplementedError(
                "Flow variance has not been implemented in pytorch - set use_probs to False"
            )

        # configure transformer
        self.transformer = SpatialTransformer(inshape)

    def forward(self, x: torch.Tensor):
        """
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        """

        # concatenate inputs and propagate unet
        source = x[:, 0:1, :, :]
        x = self.unet_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field

        # warp image with flow field
        y_source = self.transformer(source, pos_flow)

        return y_source, pos_flow
