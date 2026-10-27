"""Image transforms serialized in supported classification checkpoints."""


class CenterCrop:
    def __call__(self, image):
        import cv2
        height, width = image.shape[:2]
        size = min(height, width)
        top, left = (height-size)//2, (width-size)//2
        return cv2.resize(image[top:top+size, left:left+size], (self.w, self.h))


class ToTensor:
    def __call__(self, image):
        import numpy as np
        import torch
        tensor = torch.from_numpy(np.ascontiguousarray(image[:, :, ::-1].transpose(2, 0, 1))).to(self.device)
        return (tensor.half() if self.half else tensor.float())/255.
