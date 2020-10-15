import sys
import torch


class TensorRTPart(torch.nn.Module):
    def __init__(self, ssd_module):
        super().__init__()
        self.ssd_module = ssd_module

    def forward(self, image_nchw):
        # DeepStream likes to strip off the batch dimension in order to feed outputs for postprocessing image-by-image
        # this is no good for us, so insert an extra unit dimension for DS to strip off
        # return a tuple for consistency with future
        return (image_nchw.unsqueeze(0), )


