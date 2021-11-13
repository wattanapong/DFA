__author__ = 'Haohan Wang'

import numpy as np
from scipy import signal

def fft(img):
    return np.fft.fft2(img)


def fftshift(img):
    return np.fft.fftshift(fft(img))


def ifft(img):
    return np.fft.ifft2(img)


def ifftshift(img):
    return ifft(np.fft.ifftshift(img))


def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize/2) ** 2 + (j - imageSize/2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0

def mask_radial(img, r):
    rows, cols = img.shape
    mask = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask


def generateSmoothKernel(data, r):
    result = np.zeros_like(data)
    [k1, k2, m, n] = data.shape
    mask = np.zeros([3,3])
    for i in range(3):
        for j in range(3):
            if i == 1 and j == 1:
                mask[i,j] = 1
            else:
                mask[i,j] = r
    mask = mask
    for i in range(m):
        for j in range(n):
            result[:,:, i,j] = signal.convolve2d(data[:,:, i,j], mask, boundary='symm', mode='same')
    return result


def generateDataWithDifferentFrequencies_GrayScale(Images, r):
    Images_freq_low = []
    mask = mask_radial(np.zeros([28, 28]), r)
    for i in range(Images.shape[0]):
        fd = fftshift(Images[i, :].reshape([28, 28]))
        fd = fd * mask
        img_low = ifftshift(fd)
        Images_freq_low.append(np.real(img_low).reshape([28 * 28]))

    return np.array(Images_freq_low)

def generateDataWithDifferentFrequencies_3Channel(Images, r):
    Images_freq_low = []
    Images_freq_high = []
    mask = mask_radial(np.zeros([Images.shape[1], Images.shape[2]]), r)
    for i in range(Images.shape[0]):
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * mask
            img_low = ifftshift(fd)
            tmp[:,:,j] = np.real(img_low)
        Images_freq_low.append(tmp)
        tmp = np.zeros([Images.shape[1], Images.shape[2], 3])
        for j in range(3):
            fd = fftshift(Images[i, :, :, j])
            fd = fd * (1 - mask)
            img_high = ifftshift(fd)
            tmp[:,:,j] = np.real(img_high)
        Images_freq_high.append(tmp)

    return np.array(Images_freq_low), np.array(Images_freq_high)

def genFreq(Image, r):
    mask = mask_radial(np.zeros([Image.shape[0], Image.shape[1]]), r)

    tmp = np.zeros([Image.shape[0], Image.shape[1], 3])
    fd = np.zeros_like(tmp)
    import pdb
    pdb.set_trace()

    for j in range(3):
        fd[:, :, j] = fftshift(Image[:, :, j])
        _fd = fd[:, :, j] * mask
        img_low = ifftshift(_fd)
        tmp[:, :, j] = np.real(img_low)
    Image_freq_low = tmp

    tmp = np.zeros([Image.shape[0], Image.shape[1], 3])
    for j in range(3):
        _fd = fd[:, :, j] * (1 - mask)
        img_high = ifftshift(_fd)
        tmp[:, :, j] = np.real(img_high)
    Image_freq_high = tmp

    return Image_freq_low, Image_freq_high

def genFreqHL(Image):
    tmpH = np.zeros([Image.shape[0], Image.shape[1], 3])
    tmpL = tmpH.copy()
    for j in range(3):
        fd = fftshift(Image[:, :, j])
        _fd = fd.reshape((1, Image.shape[0]*Image.shape[1]))
        _cfd = np.imag(fd).reshape((1, Image.shape[0]*Image.shape[1]))

        # ilfd = (_fd < _fd.min()).reshape((Image.shape[0], Image.shape[1]))
        # ihfd = np.invert(ilfd)
        # lfd = fd.copy()
        # hfd = fd.copy()
        # lfd[ilfd] = 0
        # hfd[ihfd] = 0

        lfd = _fd.copy()
        hfd = _fd.copy()

        lfd[0, lfd[0, :] > _cfd[0, :]] = 0
        hfd[0, hfd[0, :] <= _cfd[0, :]] = 0
        import pdb
        pdb.set_trace()
        # lfd[0, 8 * _fd.shape[1] // 16:] = 0
        # hfd[0, : 8 * _fd.shape[1] // 16] = 0
        lfd = lfd.reshape((Image.shape[0], Image.shape[1]))
        hfd = hfd.reshape((Image.shape[0], Image.shape[1]))

        img_low = ifftshift(lfd)
        tmpL[:, :, j] = np.real(img_low)
        img_high = ifftshift(hfd)
        tmpH[:, :, j] = np.real(img_high)
    Image_freq_high = tmpH
    Image_freq_low = tmpL
    return Image_freq_low, Image_freq_high
