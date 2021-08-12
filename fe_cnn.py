import chainer
from chainer import cuda, optimizers, Variable
import chainer.functions as F
import chainer.links as L
from chainer import function_hooks


class Linear(chainer.Chain):
    def __init__(self, n_in, n_out):
        initializer = chainer.initializers.GlorotUniform()
        super(Linear, self).__init__()
        with self.init_scope():
            self.linear = L.Linear(n_in, n_out, initialW=initializer)
            self.bn = L.BatchNormalization(n_out)

    def __call__(self, x):
        h = self.linear(x)
        h = self.bn(h)

        return h

class Block(chainer.Chain):

    def __init__(self, out_channels, ksize, pad=1):
        initializer = chainer.initializers.GlorotNormal()
        super(Block, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(None, out_channels, ksize, pad=pad,
                                        nobias=True, initialW=initializer)
            self.bn = L.BatchNormalization(out_channels)

    def __call__(self, x):
        h = self.conv(x)
        h = self.bn(h)
        return F.relu(x) #F.sigmoid(h)*h

class FE(chainer.Chain):
    def __init__(self):
        super(FE, self).__init__()
        with self.init_scope():
            self.l1 = Block(64, ksize=(1, 5))
            self.l2 = Block(64, ksize=(1, 5))
            self.l3 = Block(256, ksize=(1, 5))
            self.l4 = Block(256, ksize=(1, 10))
            self.l5 = Block(512, ksize=(1, 10))
            self.l6 = Block(512, ksize=(1, 10))
            self.l7 = L.Linear(None, 512, nobias=True)
            self.bn_fc = L.BatchNormalization(512)

    def __call__(self, x):

        h = F.relu(self.l1(x))
        #h = F.max_pooling_2d(h, ksize=(1, 2))
        #h = F.dropout(h, ratio=0.7)

        h = F.relu(self.l2(h))
        h = F.max_pooling_2d(h, ksize=(1, 4))
        #h = F.dropout(h, ratio=0.7)


        h = F.relu(self.l3(h))
        #h = F.max_pooling_2d(h, ksize=(1, 2))
        #h = F.dropout(h, ratio=0.7)

        h = F.relu(self.l4(h))
        h = F.max_pooling_2d(h, ksize=(1, 4))
        #h = F.dropout(h, ratio=0.7)

        h = F.relu(self.l5(h))
        #h = F.max_pooling_2d(h, ksize=(1, 2))
        #h = F.dropout(h, ratio=0.7)

        h = F.relu(self.l6(h))
        h = F.max_pooling_2d(h, ksize=(1, 4))

        h = F.relu(self.l7(h))

        h = self.bn_fc(h)

        return h


