import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

# opencv 图像的基本运算

# 导入库
import cv2
from matplotlib import pyplot as plt
from scipy import special
from torchvision import transforms

from models.blocks import SpatialAtt, ChannelShuffle
from models.common import Conv
from utils.plots import feature_visualization


def main():
    torch.random.manual_seed(0)

    f1 = torch.randn(1, 256, 1, 1)

    module = nn.Sequential(OrderedDict(
        conv=nn.Conv2d(in_s=2, out_s=2, kernel_size=3, stride=1, padding=1, bias=False),
        bn=nn.BatchNorm2d(num_features=2)
    ))

    module.eval()

    with torch.no_grad():
        output1 = module(f1)
        print(output1)

    temp1 = module.conv[0]
    temp2 = module.conv[1]

    sumTemp = temp1 + temp2

    # fuse conv + bn
    kernel = module.conv.weight 
    running_mean = module.bn.running_mean
    running_var = module.bn.running_var
    gamma = module.bn.weight
    beta = module.bn.bias
    eps = module.bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1, 1)  # [ch] -> [ch, 1, 1, 1]
    kernel = kernel * t
    bias = beta - running_mean * gamma / std
    fused_conv = nn.Conv2d(in_s=2, out_s=2, kernel_size=3, stride=1, padding=1, bias=True)
    fused_conv.load_state_dict(OrderedDict(weight=kernel, bias=bias))

    with torch.no_grad():
        output2 = fused_conv(f1)
        print(output2)

    np.testing.assert_allclose(output1.numpy(), output2.numpy(), rtol=1e-03, atol=1e-05)
    print("convert module has been tested, and the result looks good!")

def main2():
    downsample = lambda x: x
    d = downsample(16)
    print(d)

def main3():
    torch.random.manual_seed(0)
    x = torch.randn(1, 66, 4, 4)
    y = torch.randn(1, 2, 2, 2)
    # x_wh = int(math.sqrt(x.size()[1]))
    # x1 = x.view(1, 1, x_wh, x_wh)
    x1 = torch.split(x, 66//3, 1)

    x1 = torch.sum(x, dim=1)
    y1 = torch.sum(y, dim=1)


    print(1)

def mask_main():
    # 加载图像
    image = cv2.imread('inference/0000005_00509_d_0000002.jpg')
    # cv2.imshow("image loaded", image)
    imgs1 = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)
    img1 = torch.max(imgs1, dim=0)[0].unsqueeze(0)
    img11 = nn.MaxPool2d(3, 1, 1)(img1)
    img21 = torch.mean(imgs1, dim=0).unsqueeze(0)
    imgs = Conv(3, 1, 1, 1)(imgs1.unsqueeze(0)).squeeze(0)
    img3 = (Conv(3, 1, 3, 1, 1, act=nn.Sigmoid())(torch.cat([img1.unsqueeze(0), imgs.unsqueeze(0), img21.unsqueeze(0)], 1)) * Conv(3, 3, 1, 1)(imgs1.unsqueeze(0))).squeeze(0)

    iii = nn.Conv2d(3, 32, 3, 1, 1)(imgs1.unsqueeze(0))
    iim = nn.BatchNorm2d(32)(iii)

    # iim = Conv(3, 32, 1, 1)(imgs1.unsqueeze(0)) * ChannelShuffle(32, 4)(Conv(3, 32, 3, 1, 1, act=nn.Sigmoid())(nn.MaxPool2d(3, 1, 1)(imgs1.unsqueeze(0))))

    # 创建矩形区域，填充白色255
    x = torch.zeros(image.shape[0], image.shape[1])
    y = x.cpu().numpy()
    # cv2.imshow("Original", y)
    # cv2.waitKey(0)
    y[10:20, 5:15] = 1
    # cv2.imshow("Rectangle", y)
    # cv2.waitKey(0)
    o = torch.tensor(y, dtype=torch.long)

    feature_visualization(iii, "Conv.iii", None)
    feature_visualization(iim, "Conv.iim", None)


    # plt.imshow(transforms.ToPILImage()(imgs1), interpolation="bicubic")
    # transforms.ToPILImage()(imgs1).show()  # Alternatively
    #
    # plt.imshow(transforms.ToPILImage()(img1), interpolation="bicubic")
    # transforms.ToPILImage()(img1).show('max')  # Alternatively
    #
    # plt.imshow(transforms.ToPILImage()(img11), interpolation="bicubic")
    # transforms.ToPILImage()(img11).show('max')  # Alternatively
    #
    # plt.imshow(transforms.ToPILImage()(img2), interpolation="bicubic")
    # transforms.ToPILImage()(img2).show('max')  # Alternatively
    # plt.imshow(transforms.ToPILImage()(img21), interpolation="bicubic")
    # transforms.ToPILImage()(img21).show()  # Alternatively

    # plt.imshow(transforms.ToPILImage()(img3), interpolation="bicubic")
    # transforms.ToPILImage()(img3).show()  # Alternatively

    # plt.imshow(transforms.ToPILImage()(img4), interpolation="bicubic")
    # transforms.ToPILImage()(img4).show()  # Alternatively
    print('s')

##### 夜间图像增强 #####
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('double') - min_val) / (max_val - min_val)
    return out

def changePic(X, param):
    I1 = (np.max(X) / np.log(np.max(X) + 1)) * np.log(X + 1)
    I2 = 1 - np.exp(-X)
    I3 = (I1 + I2) / (param + (I1 * I2))
    I4 = special.erf(param * np.arctan(np.exp(I3) - 0.5 * I3))
    I5 = (I4 - np.min(I4)) / (np.max(I4) - np.min(I4))
    return I5

def img_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)

def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    return dst

def main4():
    # 加载图像
    img = cv2.imread('inference/9999952_00000_d_0000178.jpg')
    out = im2double(img)
    out_img = changePic(out, 5)

    # img1 = cv2.resize(out, (500, 500))
    # img2 = cv2.resize(out_img, (500, 500))

    new_img = np.hstack([out, out_img])
    img_show('new_img', new_img)

    # out_img = out_img * 255
    # img_show("chang01.jpg", out_img)
##### 夜间图像增强 #####

# 自适应白平衡
def white_balance_1(img):
    '''
    第一种简单的求均值白平衡法
    :param img: cv2.imread读取的图片数据
    :return: 返回的白平衡结果图片数据
    '''
    # 读取图像
    r, g, b = cv2.split(img)
    r_avg = cv2.mean(r)[0]
    g_avg = cv2.mean(g)[0]
    b_avg = cv2.mean(b)[0]
    # 求各个通道所占增益
    k = (r_avg + g_avg + b_avg) / 3
    kr = k / r_avg
    kg = k / g_avg
    kb = k / b_avg
    r = cv2.addWeighted(src1=r, alpha=kr, src2=0, beta=0, gamma=0)
    g = cv2.addWeighted(src1=g, alpha=kg, src2=0, beta=0, gamma=0)
    b = cv2.addWeighted(src1=b, alpha=kb, src2=0, beta=0, gamma=0)
    balance_img = cv2.merge([b, g, r])
    return balance_img

# 高光抑制
def main5():
    # img = cv2.imread('inference/9999979_00000_d_0000025.jpg')
    imgpath = 'inference/9999979_00000_d_0000025.jpg'
    mask = create_mask(imgpath)
    dst = xiufu(imgpath, mask)
    # new_img = np.hstack([img, dst])
    out = im2double(dst)
    out_img = changePic(out, 5)
    # image = cv2.imread('inference/9999979_00000_d_0000025.jpg', cv2.IMREAD_GRAYSCALE)
    # new_img = 1.0 / (1 + math.e**(-img))
    new_img = np.hstack([out_img, dst])
    img_show('new_img', new_img)



# 找亮光位置
def create_mask(imgpath):
    image = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    return mask
# 修复图片
def xiufu(imgpath, mask):
    src_ = cv2.imread(imgpath)
    # mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
    # 缩放因子(fx,fy)
    res_ = cv2.resize(src_, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    mask = cv2.resize(mask, None, fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    dst = cv2.inpaint(res_, mask, 10, cv2.INPAINT_TELEA)
    return dst

# 直方图均衡
def hist():
    img = cv2.imread('inference/9999979_00000_d_0000025.jpg')
    (b, g, r) = cv2.split(img)  # 通道分解
    bH = cv2.equalizeHist(b)
    gH = cv2.equalizeHist(g)
    rH = cv2.equalizeHist(r)
    result = cv2.merge((bH, gH, rH), )  # 通道合成
    res = np.hstack((img, result))
    cv2.imshow('dst', res)
    cv2.waitKey(0)

if __name__ == '__main__':
    main5()
