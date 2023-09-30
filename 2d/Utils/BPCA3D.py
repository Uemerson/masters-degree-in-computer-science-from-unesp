import torch.nn as nn
import torch

class BPCA3D(nn.Module):
    """ Custom BPCA 3D """
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

        def get_dim_blocks(dim_in, dim_kernel_size, dim_padding = 0, dim_stride = 1, dim_dilation = 1):
            dim_out = (dim_in + 2 * dim_padding - dim_dilation * (dim_kernel_size - 1) - 1) // dim_stride + 1
            return dim_out

        channels = x.shape[1]

        d_dim_in = x.shape[2]
        h_dim_in = x.shape[3]
        w_dim_in = x.shape[4]
        d_dim_out = get_dim_blocks(d_dim_in, kernel_size[0], padding[0], stride[0], dilation[0])
        h_dim_out = get_dim_blocks(h_dim_in, kernel_size[1], padding[1], stride[1], dilation[1])
        w_dim_out = get_dim_blocks(w_dim_in, kernel_size[2], padding[2], stride[2], dilation[2])
        # print(d_dim_in, h_dim_in, w_dim_in, d_dim_out, h_dim_out, w_dim_out)

        # (B, C, D, H, W)
        x = x.view(-1, channels, d_dim_in, h_dim_in * w_dim_in)
        # (B, C, D, H * W)

        x = torch.nn.functional.unfold(x, kernel_size=(kernel_size[0], 1), padding=(padding[0], 0), stride=(stride[0], 1), dilation=(dilation[0], 1))
        # (B, C * kernel_size[0], d_dim_out * H * W)

        x = x.view(-1, channels * kernel_size[0] * d_dim_out, h_dim_in, w_dim_in)
        # (B, C * kernel_size[0] * d_dim_out, H, W)

        x = torch.nn.functional.unfold(x, kernel_size=(kernel_size[1], kernel_size[2]), padding=(padding[1], padding[2]), stride=(stride[1], stride[2]), dilation=(dilation[1], dilation[2]))
        # (B, C * kernel_size[0] * d_dim_out * kernel_size[1] * kernel_size[2], h_dim_out, w_dim_out)

        x = x.view(-1, channels, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)
        # (B, C, kernel_size[0], d_dim_out, kernel_size[1], kernel_size[2], h_dim_out, w_dim_out)

        x = x.permute(0, 1, 3, 6, 7, 2, 4, 5)
        # (B, C, d_dim_out, h_dim_out, w_dim_out, kernel_size[0], kernel_size[1], kernel_size[2])

        x = x.contiguous().view(-1, channels, kernel_size[0], kernel_size[1], kernel_size[2])
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
        b, c, s, w, h = x.shape
        blocks = self.extract_patches_3d(x, 
                                         kernel_size=self.kernel_size, 
                                         padding=self.padding, 
                                         stride=self.stride,
                                         dilation=self.dilation)
        d = (b*c) // (self.kernel_size * self.kernel_size * self.kernel_size)
        data = blocks.reshape(h*w*s*d, 
                                (self.kernel_size*self.kernel_size*self.kernel_size))
        pca, mean, pca_components = self.pca(data)
        return pca.reshape(b, 
                           c, 
                           s // self.kernel_size, 
                           w // self.kernel_size, 
                           h // self.kernel_size), mean, pca_components
