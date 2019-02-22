import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Variable, optimizers
import numpy as np
import matplotlib.pyplot as plt


class Network(chainer.Chain):
    def __init__(self, n_in, n_out, stride=1):
        w = chainer.initializers.HeNormal()
        super(Network, self).__init__(
            bn1=L.BatchNormalization(n_in),
            conv1=L.Convolution2D(n_in, n_out, 3, stride, 1, nobias=True, initialW=w),
            bn2=L.BatchNormalization(n_out),
            conv2=L.Convolution2D(n_out, n_out, 3, 1, 1, nobias=True, initialW=w),
            shortcut=L.Convolution2D(n_in, n_out, 1, stride, nobias=True, initialW=w)
        )
        self.train = True

    def __call__(self, x):
        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))
        shortcut = self.shortcut(x)
        return h + shortcut


class Block(chainer.ChainList):
    def __init__(self, n_in, n_out, n_blocks, stride=2):
        super(Block, self).__init__()
        self.add_link(Network(n_in, n_out, stride))
        for _ in range(n_blocks - 1):
            self.add_link(Network(n_out, n_out))

    def __call__(self, x):
        for link in self:
            x = link(x)
        return x


class WRN(chainer.Chain):
    def __init__(self, n_class=16, n_blocks=2, k_times=8):
        w = chainer.initializers.HeNormal()
        super(WRN, self).__init__(
            conv1=L.Convolution2D(3, 16, 3, 1, 1, nobias=True, initialW=w),
            wide2=Block(16, 16 * k_times, n_blocks, 1),
            wide3=Block(16 * k_times, 32 * k_times, n_blocks, 2),
            wide4=Block(32 * k_times, 64 * k_times, n_blocks, 2),
            fc6=L.Linear(64 * k_times, n_class, initialW=w)
        )

    def __call__(self, x):
        h = self.conv1(x)
        h = self.wide2(h)
        h = self.wide3(h)
        h = self.wide4(h)
        h = F.average_pooling_2d(h, (h.shape[2], h.shape[3]))
        h = self.fc6(h)
        return h


class WRN16(WRN):
    # depth = 4 + 6 * 2 = 16
    def __init__(self, n_class=16, k_times=8):
        super(WRN16, self).__init__(n_class, 2, k_times)


class WRN22(WRN):
    # depth = 4 + 6 * 3 = 22
    def __init__(self, n_class=16, k_times=8):
        super(WRN22, self).__init__(n_class, 3, k_times)


class WRN28(WRN):
    # depth = 4 + 6 * 4 = 28
    def __init__(self, n_class=16, k_times=8):
        super(WRN28, self).__init__(n_class, 4, k_times)


def main():
    # n_blocks = 2
    # depth = 4 + 6 * n_blocks
    # データセット読み込み
    img_data = np.load("dataset_np/img_data.npy")
    img_target = np.load("dataset_np/img_target.npy")
    img_data = img_data.astype(np.float32)
    img_target = img_target.astype(np.int32)

    N = 2000
    x_train, x_test = np.split(img_data, [N])
    t_train, t_test = np.split(img_target, [N])

    '''
    データの形
    x_train = N * 3 * 200 * 200
    '''
    batch_size = 1  # max size
    n_epoch = 4

    model = WRN(16, 2, 8)

    opt = optimizers.MomentumSGD()
    opt.setup(model)

    # GPU
    gpu = 0
    if gpu >= 0:
        cuda.get_device(gpu).use()
        model.to_gpu()
    xp = np if gpu < 0 else cuda.cupy

    for epoch in range(1, n_epoch + 1):
        for i in range(0, len(x_train), batch_size):
            print(str(epoch) + ":" + str(i))
            x_batch = x_train[i:i + batch_size]
            t_batch = t_train[i:i + batch_size]
            x = Variable(xp.asarray(x_batch))
            t = Variable(xp.asarray(t_batch))
            opt.zero_grads()
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            print(loss.data)
            acc = F.accuracy(y, t)
            loss.backward()
            opt.update()


if __name__ == '__main__':
    main()
