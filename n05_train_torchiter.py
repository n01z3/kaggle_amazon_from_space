import mxnet as mx
import numpy as np
import importlib
import os, sys
import logging
import random
import argparse
from sklearn.metrics import log_loss
import pandas as pd
import glob
import time
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from n06_pytorch_utils import augment
import cv2


def fbeta_score(y_true: np.ndarray, y_pred: np.ndarray, eps=1e-7) -> float:
    # same as fbeta_score(y_true, y_pred, beta=2, average='samples')
    # but faster
    tp = (y_true * y_pred).sum(axis=1)
    r = tp / y_true.sum(axis=1)
    p = tp / (y_pred.sum(axis=1) + eps)
    beta2 = 4
    f2 = (1 + beta2) * p * r / (beta2 * p + r + eps)
    return f2.mean()


def logloss_score(label, pred_prob):
    return log_loss(label.ravel(), pred_prob.ravel(), eps=1e-7)


def f2beta(label, pred_prob):
    t = 0.2
    return fbeta_score(label, pred_prob > t)


def f2beta_adaptive(y_tru, y_prd):
    ts = np.arange(0.05, 0.5, 0.01)

    scrs = [fbeta_score(y_tru, y_prd > t) for t in ts]
    bts = np.ones(17, np.float32) * ts[np.argmax(scrs)]

    for i in range(17):
        adj = np.arange(-0.1, 0.1, 0.01)

        scrs = []
        for a in adj:
            thresholds = bts.copy()
            thresholds[i] += a
            scrs.append(fbeta_score(y_tru, y_prd > thresholds))

        bts[i] += adj[np.argmax(scrs)]

    return fbeta_score(y_tru, y_prd > bts)


def check_iters(ags):
    trn, val = pytorch_data_iter(ags)

    plt.figure()

    for j, val_data in enumerate(val, 0):
        inputs, labels = val_data
        inputs, labels = inputs.numpy(), labels.numpy()

        print(inputs.shape, labels.shape, np.amax(inputs), np.amin(inputs), np.mean(inputs))
        for i in range(5):
            plt.subplot(2, 5, 1 + i)
            plt.imshow(np.transpose(inputs[i], (1, 2, 0)))
        break

    for j, trn_data in enumerate(trn, 0):
        inputs, labels = trn_data
        inputs, labels = inputs.numpy(), labels.numpy()

        print(inputs.shape, labels.shape, np.amax(inputs), np.amin(inputs), np.mean(inputs))
        for i in range(5):
            plt.subplot(2, 5, 6 + i)
            plt.imshow(np.transpose(inputs[i], (1, 2, 0)))
        break

    plt.show()


def get_mod(ags):
    bst = -1
    if ags.sfrom == 'i' or len(glob.glob(f'weights/{ags.versn}/{ags.nfold}/*json')) == 0:
        logging.info('start from imagenet weights')
        sym, arg_params, aux_params = mx.model.load_checkpoint(
            f'weights/imagenet/resnet/{ags.model}', 0)
        fea_symbol = sym.get_internals()["flatten0_output"]

        drp = mx.symbol.Dropout(data=fea_symbol, p=0.5)
        fc1 = mx.symbol.FullyConnected(data=drp, num_hidden=17, name='fc2')
        sym = mx.symbol.LogisticRegressionOutput(data=fc1, name='softmax')

    elif ags.sfrom == '-1':
        jsn = sorted(glob.glob(f'weights/{ags.versn}/{ags.nfold}/*json'))[-1]
        jsn = jsn.split('-symbol.')[0]
        logging.info(f"start from {jsn}")
        sym, arg_params, aux_params = mx.model.load_checkpoint(jsn, 0)
        bst = float(jsn.split('/')[-1])

    else:
        sym, arg_params, aux_params = mx.model.load_checkpoint(f'weights/{ags.sfrom}/{ags.nfold}/-0', ags.begin)

    dev = [mx.gpu(int(i)) for i in ags.ngpus.split(',')]

    if ags.trnfc:
        fixed_param_names = [name for name in sym.list_arguments() if
                             name not in ['fc2_weight', 'fc2_bias', 'softmax_label', 'data']]
        logging.info("Freezed parameters: [" + ','.join(fixed_param_names) + ']')
    else:
        fixed_param_names = None

    mod = mx.mod.Module(context=dev, symbol=sym, fixed_param_names=fixed_param_names)
    mod.bind(data_shapes=[('data', (ags.batch, 3, ags.isize, ags.isize))],
             label_shapes=[('softmax_label', (ags.batch, 17))])

    mod.init_params(mx.init.Uniform(scale=.1), arg_params, aux_params, allow_missing=True)
    if ags.optim == 'sgd':
        mod.init_optimizer(kvstore='local_allreduce_device', optimizer=ags.optim,
                           optimizer_params={"learning_rate": ags.lrate, "momentum": ags.momen, "wd": ags.wdecy})
    else:
        mod.init_optimizer(optimizer=ags.optim)

    return mod, bst


class CSVDataset(data.Dataset):
    def __init__(self, path, transform=None):
        df = pd.read_csv(path)

        self.path = np.array([fn.replace('code/amazon_from_space/_data', 'dataset/amazon') for fn in df.iloc[:, 0].tolist()])
        self.target = df.iloc[:, 1:].values.astype(np.float32)

        self.transform = transform

    def __len__(self):
        return self.target.shape[0]

    @staticmethod
    def _load_cv2(path):
        img = cv2.imread(path, 1)
        return img[:, :, [2, 1, 0]]

    def __getitem__(self, idx):
        X = self._load_cv2(self.path[idx])
        if self.transform:
            X = self.transform(X)
        y = self.target[idx]
        return X, y


def pytorch_data_iter(ags):
    val_dump = f'dump/{ags.versn}_{ags.nfold}_val.npz'
    if not os.path.isfile(val_dump):
        convert_val(ags, val_dump)

    tfz = np.load(val_dump)
    x, y = tfz['x'], tfz['y']
    print(x.shape, y.shape)

    val = mx.io.NDArrayIter(data={'data': x}, label={'softmax_label': y}, batch_size=ags.batch)
    trn_dataset = CSVDataset(f'../../_data/fold{ags.nfold}/train.csv', transform=augment)
    trn = data.DataLoader(trn_dataset, batch_size=ags.batch, shuffle=True, num_workers=8)
    return trn, val


def convert_val(ags, fn):
    print('start to convert val')

    def preproc(x):
        return np.transpose(x, (2, 0, 1))

    if not os.path.exists('dump'): os.mkdir('dump')
    val_dataset = CSVDataset(f'../../_data/fold{ags.nfold}/val.csv', transform=preproc)
    val = data.DataLoader(val_dataset, batch_size=ags.batch, shuffle=False, num_workers=8)

    x, y = [], []
    for i, val_data in enumerate(val):
        inputs, labels = val_data
        inputs, labels = inputs.numpy(), labels.numpy()
        print(inputs.shape, labels.shape)
        x.append(inputs)
        y.append(labels)

    x, y = np.vstack(x), np.vstack(y)
    print(x.shape, y.shape)
    np.savez(fn, x=x, y=y)


def main(ags):
    # os.environ["CUDA_VISIBLE_DEVICES"] = ags.ngpus

    for pth in ['logs', 'weights', 'predicts']:
        pth = os.path.join(pth, ags.versn)
        if not os.path.exists(pth): os.mkdir(pth)
    if not os.path.exists(f'weights/{ags.versn}/{ags.nfold}'): os.mkdir(f'weights/{ags.versn}/{ags.nfold}')

    log_file = os.path.join('logs', ags.versn, f'{ags.versn}_{ags.nfold}_log')
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(log_file))
    logging.info('mxnet version %s', mx.__version__)
    logging.info('start with arguments %s', ags)

    y_val = pd.read_csv(f'../../_data/fold{ags.nfold}/val.csv').iloc[:, 1:].values.astype(np.float32)

    mod, bst_scr = get_mod(ags)
    trn, val = pytorch_data_iter(ags)

    logging.info(f'itr f2_val ce_val f2_trn ce_trn')

    ce_trn = mx.metric.np(logloss_score)
    f2_trn = mx.metric.np(f2beta)

    cnt, icnt, ttr, tvl = 0, 0, time.time(), time.time()
    for epoch in range(ags.nepoh):
        for i, data in enumerate(trn):
            inputs, labels = data
            inputs, labels = inputs.numpy(), labels.numpy()
            if len(inputs) < ags.batch: break

            batch = mx.io.DataBatch([mx.nd.array(inputs, ctx=mx.cpu())], [mx.nd.array(labels, ctx=mx.cpu())])

            mod.forward_backward(batch)
            mod.update()
            mod.update_metric(f2_trn, batch.label)
            mod.update_metric(ce_trn, batch.label)
            icnt += 1

            if icnt % (ags.check // 5) == 0:
                logging.info(
                    f'trn {ce_trn.get()[1]} {f2_trn.get()[1]} {time.time() - ttr} {(ags.check * ags.batch // 5) / (time.time() - ttr)}')
                ce_trn.reset()
                f2_trn.reset()
                ttr = time.time()

            if icnt % ags.check == 0:
                y_prd = mod.predict(val).asnumpy()

                ce = logloss_score(y_val, y_prd)
                f2 = f2beta(y_val, y_prd)

                trg = 1.0 - ce
                t1 = time.time() - tvl
                logging.info(
                    f'val {icnt} {ce:{6}.{5}} {1-ce-bst_scr:{6}.{5}} {f2:{6}.{5}} {t1}')

                if trg < bst_scr + 0.00001:
                    cnt += 1
                else:
                    cnt = 0
                    bst_scr = trg

                mx.model.save_checkpoint(f'weights/{ags.versn}/{ags.nfold}/{trg:{5}.{4}}', 0, mod.symbol,
                                         *mod.get_params())

                if cnt > ags.patin:
                    logging.info('cancel, best score: %0.6f' % bst_scr)
                    return None

                for tp in ['json', 'params']:
                    for fn in sorted(glob.glob(f'weights/{ags.versn}/{ags.nfold}/*{tp}'))[:-5]: os.remove(fn)

                tvl = time.time()


def add_fit_args(train):
    train.add_argument('--ngpus', default='0', type=str, help='number of gpu')
    train.add_argument('--versn', default='rn50_1', type=str, help='version of net')
    train.add_argument('--begin', default=-1, type=int, help='start epoch')
    train.add_argument('--sfrom', default='-1', type=str, help='start epoch')

    train.add_argument('--model', default='resnet-50', type=str, help='model name')
    train.add_argument('--isize', default=256, type=int, help='input image size')

    train.add_argument('--batch', default=12, type=int, help='the batch size')
    train.add_argument('--trnfc', default=False, type=bool, help='the batch size')
    train.add_argument('--nepoh', default=30, type=int, help='amount of epoch')
    train.add_argument('--check', default=300, type=int, help='period of check in iteration')
    train.add_argument('--lrate', default=0.0001, type=float, help='start learning rate')
    train.add_argument('--optim', default='sgd', type=str, help='optimizer')
    train.add_argument('--momen', default=0.9, type=float, help='momentum for sgd')
    train.add_argument('--wdecy', default=0.0001, type=float, help='weight decay for sgd')

    train.add_argument('--patin', default=10, type=int, help='waiting for n iteration without improvement')
    train.add_argument('--auglv', default=1, type=int, help='enable augs')

    train.add_argument('--wpath', default='weights', type=str, help='net symbol path')
    train.add_argument('--nfold', default=0, type=int, help='data_path')

    train.add_argument('--kvstr', default='device', type=str, help='key-value store type')
    train.add_argument('--kdist', default=False, type=bool, help='Knowledge distilation')
    return train


if __name__ == '__main__':
    for pth in ['logs', 'weights', 'predicts']:
        if not os.path.exists(pth): os.mkdir(pth)

    parser = argparse.ArgumentParser()
    parser = add_fit_args(parser)
    ags = parser.parse_args()
    main(ags)
