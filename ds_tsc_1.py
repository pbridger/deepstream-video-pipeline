import sys
import torch

import ds_trt_1 as ds_trt
import ds_ssd300


class TorchScriptPart(torch.nn.Module):
    def __init__(self, ssd_module):
        super().__init__()
        self.ssd_module = ssd_module

    def forward(self, image_nchw):
        image_batch = self.ssd_module.preprocess(image_nchw)
        locs, labels = self.ssd_module.detector(image_batch)
        return self.ssd_module.postprocess(locs, labels)


