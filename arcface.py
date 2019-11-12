import math

import chainer
import chainer.functions as F


class ArcFace(chainer.Chain):
    def __init__(self, temperature=30, margin=0.5, *feature_extractors):
        super(ArcFace, self).__init__()
        
        if len(feature_extractors) > 1:  # CNNs don't share parameters.
            self.fe0, self.fe1 = feature_extractors
        else: # CNNs shares parameters.
            self.fe0 = feature_extractors[0]
            self.fe1 = self.fe0

        self.temperature = temperature  # softmax temepature
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        # Angle between two vectors must be between 0 and pi.
        # self.bound is used for the check.
        self.bound = math.cos(math.pi - margin)
        # If the angle+margin is over pi. We cannot use ArcFace.        
        # So, we use CosFace instead in that cases.
        # ArcFace: cos(theta + margin)
        # CosFace: cos(theta) - margin
        self.margin_cosface = math.sin(math.pi - margin) * margin

    def forward(self, batch_a, batch_b):
        len_batch = batch_a.shape[0]

        feature_a = self.extract_feature(batch_a, self.fe0)
        feature_b = self.extract_feature(batch_b, self.fe1)
        # feature_a.shape = (len_batch, dim_feature)

        cos = feature_a @ feature_b.T
        sin = F.sqrt(F.clip(1 - F.square(cos), 0, 1))

        # cos(theta+m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cos * self.cos_m - sin * self.sin_m 
        # In case (theta+margin)>pi, use cosface instead.
        phi = F.where(cos.array > self.bound, phi, cos - self.margin_cosface)

        # Diagonal elements are positive pairs: cos(theta+margin)).
        # Other elements are negative pairs: cos(theta)        
        xp = feature_a.xp
        identity_mat = xp.eye(len_batch, dtype=xp.bool)
        phi = self.temperature * F.where(identity_mat, phi, cos)

        # Generate positive labels
        label = xp.array(range(len_batch))
        return F.softmax_cross_entropy(phi, label)

    def extract_feature(self, batch, feature_extractor):
        return F.normalize(feature_extractor(batch), axis=0)
