import sys
import torch


class TensorRTPart(torch.nn.Module):
    def __init__(self, ssd_module):
        super().__init__()
        self.ssd_module = ssd_module
        self.creates_dummy_dim = False

    def forward(self, image_nchw):
        # return a tuple for consistency with future
        return (image_nchw.half(), ) # no-op to avoid entirely empty TRT engine

if __name__ == '__main__':
    print('image_nchw')
