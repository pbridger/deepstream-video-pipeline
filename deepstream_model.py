import math, itertools
import torch, torchvision


def init_dboxes(device, model_dtype):
    'adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/utils.py'
    fig_size = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    fk = fig_size / torch.tensor(steps).float()

    dboxes = []
    # size of feature and number of feature
    for idx, sfeat in enumerate(feat_size):
        sk1 = scales[idx] / fig_size
        sk2 = scales[idx + 1] / fig_size
        sk3 = math.sqrt(sk1 * sk2)
        all_sizes = [(sk1, sk1), (sk3, sk3)]

        for alpha in aspect_ratios[idx]:
            w, h = sk1 * math.sqrt(alpha), sk1 / math.sqrt(alpha)
            all_sizes.append((w, h))
            all_sizes.append((h, w))

        for w, h in all_sizes:
            for i, j in itertools.product(range(sfeat), repeat=2):
                cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                dboxes.append((cx, cy, w, h))

    return torch.tensor(
        dboxes,
        dtype=model_dtype,
        device=device
    ).clamp(0, 1)


class FullSSD300(torch.nn.Module):
    def __init__(self, detection_threshold, model_precision, device):
        super().__init__()
        self.device = device
        self.detector = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=model_precision).eval().to(device)
        self.detection_threshold = 0.4
        self.model_dtype = torch.float16 if model_precision == 'fp16' else torch.float32
        self.dboxes_xywh = init_dboxes(device, self.model_dtype).unsqueeze(dim=0)
        self.scale_xy = 0.1
        self.scale_wh = 0.2

    def forward(self, X):
        image_batch = self.preprocess(X)
        locs, labels = self.detector(image_batch)
        return self.postprocess(locs, labels)

    def preprocess(self, image_batch_nhwc):
        'normalize, NHWC -> NCHW'
        return self.normalize(
            image_batch.to(self.device) / 255
        ).permute(0, 3, 1, 2).contiguous()

    def normalize(self, input_tensor):
        'Nvidia SSD300 code uses mean and std-dev of 128/256'
        return (2.0 * input_tensor) - 1.0

    def xywh_to_xyxy(self, bboxes_batch, scores_batch):
        bboxes_batch = bboxes_batch.permute(0, 2, 1)
        scores_batch = scores_batch.permute(0, 2, 1)
        return bboxes_batch, torch.nn.functional.softmax(scores_batch, dim=-1)

        bboxes_batch[:, :, :2] = self.scale_xy * bboxes_batch[:, :, :2]
        bboxes_batch[:, :, 2:] = self.scale_wh * bboxes_batch[:, :, 2:]

        bboxes_batch[:, :, :2] = bboxes_batch[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_batch[:, :, 2:] = bboxes_batch[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # transform format to ltrb
        l, t, r, b = bboxes_batch[:, :, 0] - 0.5 * bboxes_batch[:, :, 2],\
                     bboxes_batch[:, :, 1] - 0.5 * bboxes_batch[:, :, 3],\
                     bboxes_batch[:, :, 0] + 0.5 * bboxes_batch[:, :, 2],\
                     bboxes_batch[:, :, 1] + 0.5 * bboxes_batch[:, :, 3]

        bboxes_batch[:, :, 0] = l
        bboxes_batch[:, :, 1] = t
        bboxes_batch[:, :, 2] = r
        bboxes_batch[:, :, 3] = b

        return bboxes_batch, torch.nn.functional.softmax(scores_batch, dim=-1)

    def postprocess(self, locs, labels):
        locs, probs = self.xywh_to_xyxy(locs, labels)

        # flatten batch and classes
        batch_dim, box_dim, class_dim = probs.size()
        # flat_locs = locs.view(-1, 4).repeat_interleave(class_dim.cuda(), dim=0)
        flat_box_dim = batch_dim * box_dim
        flat_locs = locs.view(flat_box_dim, 4, 1)
        flat_locs = flat_locs.expand(flat_box_dim, 4, class_dim)
        flat_locs = flat_locs.flatten(1, 2)
        flat_locs = flat_locs.view(flat_box_dim * class_dim, 4)

        flat_probs = probs.view(-1)
        class_indexes = torch.arange(class_dim, device=self.device).repeat(batch_dim * box_dim)
        image_indexes = (torch.ones(box_dim * class_dim, device=self.device) * torch.arange(1, batch_dim + 1, device=self.device).unsqueeze(-1)).view(-1)

        # only do NMS on detections over threshold, and ignore background (0)
        threshold_mask = (flat_probs > self.detection_threshold) & (class_indexes > 0)
        flat_locs = flat_locs[threshold_mask]
        flat_probs = flat_probs[threshold_mask]
        class_indexes = class_indexes[threshold_mask]
        image_indexes = image_indexes[threshold_mask]

        nms_mask = torchvision.ops.boxes.batched_nms(
            flat_locs,
            flat_probs,
            class_indexes * image_indexes,
            iou_threshold=0.7
        )

        bboxes = flat_locs[nms_mask]
        probs = flat_probs[nms_mask]
        class_indexes = class_indexes[nms_mask]
        return bboxes, probs, class_indexes


if __name__ == '__main__':
    a = torch.tensor([
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [4, 4, 4, 4],
        [5, 5, 5, 5],
    ])

#     print(a.view(5, 4, 1))
#     print(a.repeat_interleave(2, dim=0))
#     print(a.unsqueeze(-1).expand(5, 4, 2).flatten(-2, -1).view(10, 4))

    image_batch = torch.randn((1, 300, 300, 3)).cuda()
    print(image_batch.size())

    model = FullSSD300(0.4, 'fp32', torch.device('cuda'))

    bboxes, *others = model(image_batch)
    print(bboxes.size())

    torch.onnx.export(
        model,
        image_batch,
        'ssd300.onnx',
        # input_names=['image_batch'],
        # output_names=['locs', 'probs'],
        opset_version=12
    )


