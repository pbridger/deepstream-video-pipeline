import sys
import torch


class TensorRTPart(torch.nn.Module):
    def __init__(self, ssd_module):
        super().__init__()
        self.ssd_module = ssd_module

    def forward(self, image_nchw):
        image_batch = self.ssd_module.preprocess(image_nchw)
        locs, labels = self.ssd_module.detector(image_batch)
        boxes, probs, classes = self.ssd_module.postprocess(locs, labels)
        # DeepStream likes to strip off the batch dimension in order to feed outputs for postprocessing image-by-image
        # this is no good for us, so insert an extra unit dimension for DS to strip off
        # return a tuple for consistency with future
        return (boxes.unsqueeze(0), probs.unsqueeze(0), classes.unsqueeze(0))

if __name__ == '__main__':
    print('boxes,probs,classes')
