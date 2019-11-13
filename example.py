from pathlib import Path

import chainer
import chainer.links as L
import numpy as np
from chainer.iterators import MultiprocessIterator
from chainer.optimizers import Adam
from chainer.training import Trainer
from chainer.training.updaters import StandardUpdater
from chainercv.links.model.resnet import ResNet50

from models.arcface import ArcFace
from paired_image_dataset import PairedImageSet

chainer.config.cv_resize_backend = 'cv2'


if __name__ == "__main__":
    photo_path = sorted(Path('photos').glob('*'))
    sketch_path = sorted(Path('sketches').glob('*'))
    pair_list = [[str(i), str(j)] for i, j in zip(photo_path, sketch_path)]

    img_size = (200, 250)
    dataset = PairedImageSet(pair_list, '', img_size, False, np.float32)
    iter_train = MultiprocessIterator(dataset, 5, n_processes=2)
    adam = Adam(alpha=0.002, beta1=0.0, beta2=0.9)

    resnet = ResNet50(pretrained_model='imagenet')
    fc_dim = 500
    resnet.fc6 = L.Linear(None, fc_dim)  # change the number of fc layer to 500

    temp = 30
    margin = 0.5
    arcface = ArcFace(temp, margin, resnet)

    adam.setup(arcface)
    updater = StandardUpdater(iter_train, adam)
    trainer = Trainer(updater, (1000, 'iteration'))

    trainer.run()
