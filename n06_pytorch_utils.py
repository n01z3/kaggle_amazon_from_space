SEED = 666

import os

os.environ['HOME'] = '/home/cnn'
# os.environ['PYTHONUNBUFFERED'] = '1'

# numerical libs
import math
import numpy as np
import random
import PIL
import cv2

random.seed(SEED)
np.random.seed(SEED)

import matplotlib

matplotlib.use('TkAgg')

from torch.utils.data.sampler import *


def augment2(x):
    x = x.astype(np.float32) / 255
    u = 0.75
    if random.random() < u:
        if random.random() > 0.5:
            x = randomDistort1(x, distort_limit=0.35, shift_limit=0.25, u=1)
        else:
            x = randomDistort2(x, num_steps=10, distort_limit=0.2, u=1)
        x = randomShiftScaleRotate(x, shift_limit=0.0625, scale_limit=0.10, rotate_limit=45, u=1)

    x = randomFlip(x, u=0.5)
    x = randomTranspose(x, u=0.5)
    x = randomContrast(x, limit=0.2, u=0.5)
    # x = randomSaturation(x, limit=0.2, u=0.5),
    x = randomFilter(x, limit=0.5, u=0.2)
    x = 255.0 * x
    x[:, :, 0] -= 124
    x[:, :, 1] -= 117
    x[:, :, 2] -= 104

    x *= 0.0167
    x = np.transpose(x, (2, 0, 1))
    return x


def augment(x):
    x = x.astype(np.float32) / 255
    u = 0.75
    if random.random() < u:
        if random.random() > 0.5:
            x = randomDistort1(x, distort_limit=0.35, shift_limit=0.25, u=1)
        else:
            x = randomDistort2(x, num_steps=10, distort_limit=0.2, u=1)
        x = randomShiftScaleRotate(x, shift_limit=0.0625, scale_limit=0.10, rotate_limit=45, u=1)

    x = randomFlip(x, u=0.5)
    x = randomTranspose(x, u=0.5)
    x = randomContrast(x, limit=0.2, u=0.5)
    # x = randomSaturation(x, limit=0.2, u=0.5),
    x = randomFilter(x, limit=0.5, u=0.2)
    x = np.uint8(255.0 * np.transpose(x, (2, 0, 1)))
    return x


# draw -----------------------------------
def im_show(name, image, resize=1):
    H, W = image.shape[0:2]
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image.astype(np.uint8))
    cv2.resizeWindow(name, round(resize * W), round(resize * H))


def draw_shadow_text(img, text, pt, fontScale, color, thickness, color1=None, thickness1=None):
    if color1 is None: color1 = (0, 0, 0)
    if thickness1 is None: thickness1 = thickness + 2

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pt, font, fontScale, color1, thickness1, cv2.LINE_AA)
    cv2.putText(img, text, pt, font, fontScale, color, thickness, cv2.LINE_AA)


## custom data transform  -----------------------------------
def tensor_to_img(img, mean=0, std=1, dtype=np.uint8):
    img = img.numpy()
    img = np.transpose(img, (1, 2, 0))
    img = img * std + mean
    img = img.astype(dtype)
    # img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    return img


## transform (input is numpy array, read in by cv2)
def img_to_tensor(img, mean=0, std=1.):
    img = img.astype(np.float32)
    img = (img - mean) / std
    img = img.transpose((2, 0, 1))
    tensor = torch.from_numpy(img)  ##.float()
    return tensor


## for debug
def dummy_transform1(image):
    print('\t\tdummy_transform1')
    return image


def dummy_transform2(image):
    print('\t\tdummy_transform2')
    return image


# http://enthusiaststudent.blogspot.jp/2015/01/horizontal-and-vertical-flip-using.html
# http://qiita.com/supersaiakujin/items/3a2ac4f2b05de584cb11
def randomVerticalFlip(img, u=0.5):
    if random.random() < u:
        img = cv2.flip(img, 0)  # np.flipud(img)  #cv2.flip(img,0) ##up-down
    return img


def randomHorizontalFlip(img, u=0.5):
    shape = img.shape
    if random.random() < u:
        img = cv2.flip(img, 1)  # np.fliplr(img)  #cv2.flip(img,1) ##left-right
    return img


def randomFlip(img, u=0.5):
    if random.random() < u:
        img = cv2.flip(img, random.randint(-1, 1))
    return img


def randomTranspose(img, u=0.5):
    if random.random() < u:
        img = img.transpose(1, 0, 2)  # cv2.transpose(img)
    return img


# http://stackoverflow.com/questions/16265673/rotate-image-by-90-180-or-270-degrees
def randomRotate90(img, u=0.25):
    if random.random() < u:
        angle = random.randint(1, 3) * 90
        if angle == 90:
            img = img.transpose(1, 0, 2)  # cv2.transpose(img)
            img = cv2.flip(img, 1)
            # return img.transpose((1,0, 2))[:,::-1,:]
        elif angle == 180:
            img = cv2.flip(img, -1)
            # return img[::-1,::-1,:]
        elif angle == 270:
            img = img.transpose(1, 0, 2)  # cv2.transpose(img)
            img = cv2.flip(img, 0)
            # return  img.transpose((1,0, 2))[::-1,:,:]
    return img


def randomRotate(img, u=0.25, limit=90):
    if random.random() < u:
        angle = random.uniform(-limit, limit)  # degree

        height, width = img.shape[0:2]
        mat = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        img = cv2.warpAffine(img, mat, (height, width), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        # img = cv2.warpAffine(img, mat, (height,width),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

    return img


def randomShift(img, u=0.25, limit=4):
    if random.random() < u:
        dx = round(random.uniform(-limit, limit))  # pixel
        dy = round(random.uniform(-limit, limit))  # pixel

        height, width, channel = img.shape
        img1 = cv2.copyMakeBorder(img, limit + 1, limit + 1, limit + 1, limit + 1, borderType=cv2.BORDER_REFLECT_101)
        y1 = limit + 1 + dy
        y2 = y1 + height
        x1 = limit + 1 + dx
        x2 = x1 + width
        img = img1[y1:y2, x1:x2, :]

    return img


def randomShiftScale(img, u=0.25, limit=4):
    if random.random() < u:
        height, width, channel = img.shape
        assert (width == height)
        size0 = width
        size1 = width + 2 * limit
        img1 = cv2.copyMakeBorder(img, limit, limit, limit, limit, borderType=cv2.BORDER_REFLECT_101)
        size = round(random.uniform(size0, size1))

        dx = round(random.uniform(0, size1 - size))  # pixel
        dy = round(random.uniform(0, size1 - size))

        y1 = dy
        y2 = y1 + size
        x1 = dx
        x2 = x1 + size

        if size == size0:
            img = img1[y1:y2, x1:x2, :]
        else:
            img = cv2.resize(img1[y1:y2, x1:x2, :], (size0, size0), interpolation=cv2.INTER_LINEAR)

    return img


def randomShiftScaleRotate(img, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, u=0.5):
    if random.random() < u:
        height, width, channel = img.shape

        angle = random.uniform(-rotate_limit, rotate_limit)  # degree
        scale = random.uniform(1 - scale_limit, 1 + scale_limit)
        dx = round(random.uniform(-shift_limit, shift_limit)) * width
        dy = round(random.uniform(-shift_limit, shift_limit)) * height

        cc = math.cos(angle / 180 * math.pi) * (scale)
        ss = math.sin(angle / 180 * math.pi) * (scale)
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        img = cv2.warpPerspective(img, mat, (width, height), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT_101)  # cv2.BORDER_CONSTANT, borderValue = (0, 0, 0))  #cv2.BORDER_REFLECT_101

    return img


def cropCenter(img, height, width):
    h, w, c = img.shape
    dx = (h - height) // 2
    dy = (w - width) // 2

    y1 = dy
    y2 = y1 + height
    x1 = dx
    x2 = x1 + width
    img = img[y1:y2, x1:x2, :]

    return img


## unconverntional augmnet ################################################################################3
## https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion

## https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
## https://stackoverflow.com/questions/2477774/correcting-fisheye-distortion-programmatically
## http://www.coldvision.io/2017/03/02/advanced-lane-finding-using-opencv/

## barrel\pincushion distortion
def randomDistort1(img, distort_limit=0.35, shift_limit=0.25, u=0.5):
    if random.random() < u:
        height, width, channel = img.shape

        # debug
        # img = img.copy()
        # for x in range(0,width,10):
        #     cv2.line(img,(x,0),(x,height),(1,1,1),1)
        # for y in range(0,height,10):
        #     cv2.line(img,(0,y),(width,y),(1,1,1),1)

        k = random.uniform(-distort_limit, distort_limit) * 0.00001
        dx = random.uniform(-shift_limit, shift_limit) * width
        dy = random.uniform(-shift_limit, shift_limit) * height

        # map_x, map_y = cv2.initUndistortRectifyMap(intrinsics, dist_coeffs, None, None, (width,height),cv2.CV_32FC1)
        # https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion
        # https://stackoverflow.com/questions/10364201/image-transformation-in-opencv
        x, y = np.mgrid[0:width:1, 0:height:1]
        x = x.astype(np.float32) - width / 2 - dx
        y = y.astype(np.float32) - height / 2 - dy
        theta = np.arctan2(y, x)
        d = (x * x + y * y) ** 0.5
        r = d * (1 + k * d * d)
        map_x = r * np.cos(theta) + width / 2 + dx
        map_y = r * np.sin(theta) + height / 2 + dy

        img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return img


# http://pythology.blogspot.sg/2014/03/interpolation-on-regular-distorted-grid.html
## grid distortion
def randomDistort2(img, num_steps=10, distort_limit=0.2, u=0.5):
    if random.random() < u:
        height, width, channel = img.shape

        x_step = width // num_steps
        xx = np.zeros(width, np.float32)
        prev = 0
        for x in range(0, width, x_step):
            start = x
            end = x + x_step
            if end > width:
                end = width
                cur = width
            else:
                cur = prev + x_step * (1 + random.uniform(-distort_limit, distort_limit))

            xx[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        y_step = height // num_steps
        yy = np.zeros(height, np.float32)
        prev = 0
        for y in range(0, height, y_step):
            start = y
            end = y + y_step
            if end > width:
                end = height
                cur = height
            else:
                cur = prev + y_step * (1 + random.uniform(-distort_limit, distort_limit))

            yy[start:end] = np.linspace(prev, cur, end - start)
            prev = cur

        map_x, map_y = np.meshgrid(xx, yy)
        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)
        img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return img


## blur sharpen, etc
def randomFilter(img, limit=0.5, u=0.5):
    if random.random() < u:
        height, width, channel = img.shape

        alpha = limit * random.uniform(0, 1)

        ##kernel = np.ones((5,5),np.float32)/25
        kernel = np.ones((3, 3), np.float32) / 9 * 0.2

        # type = random.randint(0,1)
        # if type==0:
        #     kernel = np.ones((3,3),np.float32)/9*0.2
        # if type==1:
        #     kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])*0.5

        # kernel = alpha *sharp +(1-alpha)*blur
        # kernel = np.random.randn(5, 5)
        # kernel = kernel/np.sum(kernel*kernel)**0.5

        img = alpha * cv2.filter2D(img, -1, kernel) + (1 - alpha) * img
        img = np.clip(img, 0., 1.)

    return img


##https://github.com/pytorch/vision/pull/27/commits/659c854c6971ecc5b94dca3f4459ef2b7e42fb70
## color augmentation

# brightness, contrast, saturation-------------
# from mxnet code, see: https://github.com/dmlc/mxnet/blob/master/python/mxnet/image.py

# def to_grayscle(img):
#     blue  = img[:,:,0]
#     green = img[:,:,1]
#     red   = img[:,:,2]
#     grey = 0.299*red + 0.587*green + 0.114*blue
#     return grey


def randomBrightness(img, limit=0.2, u=0.5):
    if random.random() < u:
        alpha = 1.0 + limit * random.uniform(-1, 1)
        img = alpha * img
        img = np.clip(img, 0., 1.)
    return img


def randomContrast(img, limit=0.3, u=0.5):
    if random.random() < u:
        alpha = 1.0 + limit * random.uniform(-1, 1)

        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = img * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha * img + gray
        img = np.clip(img, 0., 1.)
    return img


def randomSaturation(img, limit=0.3, u=0.5):
    if random.random() < u:
        alpha = 1.0 + limit * random.uniform(-1, 1)

        coef = np.array([[[0.114, 0.587, 0.299]]])
        gray = img * coef
        gray = np.sum(gray, axis=2, keepdims=True)
        img = alpha * img + (1.0 - alpha) * gray
        img = np.clip(img, 0., 1.)

    return img


# return fix data for debug #####################################################3
class FixedSampler(Sampler):
    def __init__(self, data, list):
        self.num_samples = len(list)
        self.list = list

    def __iter__(self):
        # print ('\tcalling Sampler:__iter__')
        return iter(self.list)

    def __len__(self):
        # print ('\tcalling Sampler:__len__')
        return self.num_samples


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))


    # test transform
    def augment(x, u=0.5):
        if random.random() < u:
            if random.random() > 0.5:
                x = randomDistort1(x, distort_limit=0.35, shift_limit=0.25, u=1)
            else:
                x = randomDistort2(x, num_steps=10, distort_limit=0.2, u=1)

            x = randomShiftScaleRotate(x, shift_limit=0.0625, scale_limit=0.10, rotate_limit=45, u=0.5)
            x = randomFlip(x, u=0.5)
            x = randomTranspose(x, u=0.5)
            x = randomContrast(x, limit=0.2, u=0.5)
            # x = randomSaturation(x, limit=0.2, u=0.5),
        return x


    height, width = 256, 256
    for n in range(10000):
        img = np.zeros((height, width, 3), np.uint8)
        img = cv2.imread('//home/aakuzin/kaggle/dataset/amazon/train-jpg/train_10059.jpg',
                         1)  ##train_10052

        img = img.astype(np.float32) / 255
        if 0:
            for x in range(0, width, 10):
                cv2.line(img, (x, 0), (x, height), (0, 0.5, x / width), 1)

            for y in range(0, height, 10):
                cv2.line(img, (0, y), (width, y), (y / height, 0.5, 0), 1)

        # img=randomShiftScaleRotate(img,u=1)
        # img=randomDistort1(img,u=1)
        # img=randomDistort2(img,u=1)
        img = augment(img, u=1)
        im_show('img', img * 255)
        cv2.waitKey(500)  # 12000
