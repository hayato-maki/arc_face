from chainer.dataset import dataset_mixin
import numpy as np
from PIL import Image
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def _normalize(img):
    return img * 2 / 255 - 1


class PairedImageSet(dataset_mixin.DatasetMixin):
    def __init__(self,
                 file_path,  # list of list
                 dir_image,
                 image_size=(256, 256),
                 normalize=False,
                 dtype=np.float32):

        self.path = file_path
        self.dir_img = dir_image
        self.dtype = dtype
        self.img_size = image_size
        self.normalize = normalize

    def __len__(self):
        return len(self.path)

    def _load_image(self, i, j):
        full_path = os.path.join(self.dir_img, self.path[i][j])
        img = Image.open(full_path).convert('RGB').resize(self.img_size)
        return np.asarray(img, self.dtype).transpose(2, 0, 1)

    def get_example(self, i):
        img0 = self._load_image(i, 0)
        img1 = self._load_image(i, 1)

        if self.normalize:
            img0 = _normalize(img0)
            img1 = _normalize(img1)

        return img0, img1
