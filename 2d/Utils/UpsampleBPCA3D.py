import torch.nn as nn
import torch

class UpsampleBPCA3D(nn.Module):
    """ Custom Upsample BPCA 3D """
    def __init__(self, kernel_size=2, stride=2, padding=0, dilation=1, n_components=1):
        super(UpsampleBPCA3D, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.n_components = n_components


    def combine_patches_3d(self, x, kernel_size, output_shape, padding=0, stride=1, dilation=1):
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(padding, int):
            padding = (padding, padding, padding)
        if isinstance(stride, int):
            stride = (stride, stride, stride)
        if isinstance(dilation, int):
            dilation = (dilation, dilation, dilation)

        def get_dim_blocks(dim_in, dim_kernel_size, dim_padding = 0, dim_stride = 1, dim_dilation = 1):
            dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
            return dim_out

        channels = x.shape[1]
        d_dim_out, h_dim_out, w_dim_out = output_shape[2:]
        d_dim_in = get_dim_blocks(d_dim_out, kernel_size[0], padding[0], stride[0], dilation[0])
        h_dim_in = get_dim_blocks(h_dim_out, kernel_size[1], padding[1], stride[1], dilation[1])
        w_dim_in = get_dim_blocks(w_dim_out, kernel_size[2], padding[2], stride[2], dilation[2])
        # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

        x = x.view(-1, channels, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])
        # (B, C, d_dim_in, h_dim_in, w_dim_in, kernel_size[0], kernel_size[1], kernel_size[2])

        x = x.permute(0, 1, 5, 2, 6, 7, 3, 4)
        # (B, C, kernel_size[0], d_dim_in, kernel_size[1], kernel_size[2], h_dim_in, w_dim_in)

        x = x.contiguous().view(-1, channels * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)
        # (B, C * kernel_size[0] * d_dim_in * kernel_size[1] * kernel_size[2], h_dim_in * w_dim_in)

        x = torch.nn.functional.fold(x, output_size=(h_dim_out, w_dim_out), kernel_size=(kernel_size[1], kernel_size[2]), padding=(padding[1], padding[2]), stride=(stride[1], stride[2]), dilation=(dilation[1], dilation[2]))
        # (B, C * kernel_size[0] * d_dim_in, H, W)

        x = x.view(-1, channels * kernel_size[0], d_dim_in * h_dim_out * w_dim_out)
        # (B, C * kernel_size[0], d_dim_in * H * W)

        x = torch.nn.functional.fold(x, output_size=(d_dim_out, h_dim_out * w_dim_out), kernel_size=(kernel_size[0], 1), padding=(padding[0], 0), stride=(stride[0], 1), dilation=(dilation[0], 1))
        # (B, C, D, H * W)

        x = x.view(-1, channels, d_dim_out, h_dim_out, w_dim_out)
        # (B, C, D, H, W)

        return x


    def forward(self, x, output_shape, mean, pca_components):
        b, c, s, w, h = x.shape
        data = x.reshape(b*c*s*w*h, self.n_components)

        # Revert PCA transformation
        reverted_patches = torch.matmul(data, pca_components.t())

        # Add back the mean
        reverted_patches = reverted_patches + mean
        
        reverted_patches = reverted_patches.reshape(b, 
                           c, 
                           s * self.kernel_size, 
                           w * self.kernel_size, 
                           h * self.kernel_size)

        combine_data = self.combine_patches_3d(
            x=reverted_patches, 
            kernel_size=self.kernel_size,
            output_shape=output_shape, 
            padding=self.padding, 
            stride=self.stride)
        
        return combine_data

