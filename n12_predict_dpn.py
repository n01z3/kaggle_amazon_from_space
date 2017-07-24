import mxnet as mx
import numpy as np
import importlib
import os, sys
import logging
import random
import argparse

import pandas as pd
import glob
import time
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from n06_pytorch_utils import augment, randomShiftScaleRotate, randomFlip, randomTranspose, randomDistort1
from n05_train_torchiter import logloss_score, f2beta
import cv2
from collections import namedtuple
from scipy.stats import gmean
from tqdm import tqdm, trange

Batch = namedtuple('Batch', ['data'])

BATCH = 7


def get_models(versn, nfold, dev):
    out = []
    jsns = sorted(glob.glob(f'weights/{versn}/{nfold}/*json'))[::-1]
    for jsn in jsns[:3]:
        jsn = jsn.split('-symbol.')[0]
        print(f"start from {jsn}")
        sym, arg_params, aux_params = mx.model.load_checkpoint(jsn, 0)

        mod = mx.mod.Module(symbol=sym, context=dev)
        mod.bind(data_shapes=[('data', (BATCH, 3, 256, 256))], for_training=False, label_shapes=mod._label_shapes)
        mod.set_params(arg_params, aux_params)

        out.append(mod)

    return out


def augment_tst(x):
    x = x.astype(np.float32) / 255.0
    x = randomDistort1(x, distort_limit=0.35, shift_limit=0.25, u=1)
    x = randomShiftScaleRotate(x, shift_limit=0.03, scale_limit=0.05, rotate_limit=20, u=1)

    x = randomFlip(x, u=0.5)
    x = randomTranspose(x, u=0.5)

    x = np.uint8(255.0 * x)

    return x


class CSVDataset_tst(data.Dataset):
    def __init__(self, paths):
        self.path = paths
        self.target = np.zeros(len(paths))

    def __len__(self):
        return self.target.shape[0]

    @staticmethod
    def get_sample(path):
        img = cv2.imread(path, 1)
        # print(img.shape, np.amax(img), np.amin(img))
        img = img[:, :, [2, 1, 0]]

        img = np.float32(img)
        img[:, :, 0] -= 124
        img[:, :, 1] -= 117
        img[:, :, 2] -= 104
        img *= 0.0167

        lst = [img]
        lst.append(cv2.flip(img.transpose(1, 0, 2), 1))
        lst.append(cv2.flip(img, -1))
        lst.append(cv2.flip(img.transpose(1, 0, 2), 0))
        lst.append(img.transpose(1, 0, 2))
        lst.append(img[::-1])
        lst.append(img[:, ::-1])
        #
        # for i in range(BATCH - 6):
        #     lst.append(augment_tst(img))

        lst = np.array(lst)
        return np.transpose(lst, (0, 3, 1, 2))

    def __getitem__(self, idx):
        X = self.get_sample(self.path[idx])
        y = self.target[idx]
        return X, y


def check_aug():
    nfold = 0
    tst_dataset = CSVDataset_tst(f'../../_data/fold{nfold}/train.csv')
    tst = data.DataLoader(tst_dataset, batch_size=1, shuffle=False, num_workers=8)

    for j, val_data in enumerate(tst, 0):
        if j == 3:
            inputs, labels = val_data
            inputs, labels = inputs.numpy()[0], labels.numpy()[0]

            print(inputs.shape, labels.shape, np.amax(inputs), np.amin(inputs), np.mean(inputs))
            for i in range(13):
                plt.subplot(3, 5, 1 + i)
                plt.imshow(np.transpose(inputs[i], (1, 2, 0)))
            break
    plt.show()


def predict_fold(nfold, versn='rnx50_4', dev=mx.gpu(), datas='val'):
    if not os.path.exists(f'dump/{versn}'): os.mkdir(f'dump/{versn}')

    mods = get_models(versn, nfold, dev)

    if datas == 'val':
        df = pd.read_csv(f'../../_data/fold{nfold}/val.csv')
        paths = np.array(
            [fn.replace('code/amazon_from_space/_data', 'dataset/amazon') for fn in df.iloc[:, 0].tolist()])
    else:
        paths = sorted(glob.glob('../../../../dataset/amazon/test-jpg/*'))

    val_dataset = CSVDataset_tst(paths)
    val = data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    centr = []

    plt.figure()
    buf = np.zeros((3, 17))

    out = []
    gen = enumerate(val, 0)
    gen = tqdm(gen, total=len(val))

    for j, val_data in gen:
        inputs, labels = val_data
        inputs = inputs.numpy()

        for i, mod in enumerate(mods):
            mod.forward(Batch([mx.nd.array(inputs[0])]))
            y_prd = mod.get_outputs()[0].asnumpy()
            if i == 0: centr.append(y_prd[0])

            y_prd = gmean(y_prd, axis=0)

            buf[i] = y_prd

        a = gmean(buf, axis=0)
        out.append(a)

    np.save(f'dump/{versn}/{datas}_{versn}_{nfold}_21crop', np.array(out))
    np.save(f'dump/{versn}/{datas}_{versn}_{nfold}_1crop', np.array(centr))
    np.save(f'dump/{versn}/{datas}_{versn}_{nfold}_fns', np.array(paths))


def compare():
    nfold = 1
    df = pd.read_csv(f'../../_data/fold{nfold}/val.csv')
    y_val = df.iloc[:, 1:].values.astype(np.float32)

    y_1 = np.load(f'dump/dpn98_1/val_dpn98_1_{nfold}_1crop.npy')
    y_21 = np.load(f'dump/dpn98_1/val_dpn98_1_{nfold}_21crop.npy')

    print(y_21.shape)

    print(logloss_score(y_val, y_1))
    print(logloss_score(y_val, y_21))

    print('1crop:', f2beta(y_val, y_1))
    print('21crop:', f2beta(y_val, y_21))


def predict_folds(ags):
    for i in range(ags.start, ags.finsh):
        predict_fold(i, ags.versn, mx.gpu(ags.ngpus), ags.datas)


def add_fit_args(train):
    train.add_argument('--ngpus', default=0, type=int, help='number of gpu')
    train.add_argument('--versn', default='dpn98_1', type=str, help='version of net')
    train.add_argument('--start', default=8, type=int, help='start fold')
    train.add_argument('--finsh', default=10, type=int, help='finish fold ')
    train.add_argument('--datas', default='val', type=str, help='finish fold ')
    return train


def check_dumps():
    fns = glob.glob('dump/val*fns*')
    for fn in fns:
        paths = np.load(fn)
        print(paths)


if __name__ == '__main__':
    for pth in ['dumps']:
        if not os.path.exists(pth): os.mkdir(pth)

    parser = argparse.ArgumentParser()
    parser = add_fit_args(parser)
    ags = parser.parse_args()
    predict_folds(ags)
    # compare()
    # check_dumps()
