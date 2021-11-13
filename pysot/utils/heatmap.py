def maprpn(img, name, m=0, u=0,savedir=None):
    import cv2
    import pdb
    import numpy as np
    _img = img.data.cpu().numpy()[0]

    n, _h, _w = _img.shape
    for i in range(0,n):
        _x1 = _img[i]
        # m = np.median(_x1)
        # m = np.unique(_x1)
        if m != 0:
            _x1[_x1 < m] = m
        if u != 0:
            _x1[_x1 > u] = u

        _x1 = cv2.resize(_x1[ :, :], (255, 255), interpolation=cv2.INTER_AREA)

        _x1 = cv2.normalize(_x1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        _x1 = cv2.applyColorMap(_x1, cv2.COLORMAP_JET)

        if savedir == None:
            cv2.imwrite('/media/wattanapongsu/4T/temp/'+ name + '_' + str(i) +'.jpg', _x1)
        else:
            cv2.imwrite(savedir + '/' + name + '_' + str(i) + '.jpg', _x1)

def maprpn_id(img,name,id=0, m=0):
    import cv2
    import pdb
    import numpy as np
    import torch
    # if isinstance(img, torch.Tensor):
    #     _img = img.data.cpu().numpy()[id]
    # else:
    #     _img = img
    _img = img.data.cpu().numpy()[id]

    n, _h, _w = _img.shape
    for i in range(0,n):
        _x1 = _img[i]
        # m = np.median(_x1)
        # m = np.unique(_x1)
        _x1[_x1 < m] = 0

        _x1 = cv2.resize(_x1, (200, 200), interpolation=cv2.INTER_AREA)

        _x1 = cv2.normalize(_x1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        _x1 = cv2.applyColorMap(_x1, cv2.COLORMAP_JET)

        cv2.imwrite('/media/wattanapongsu/4T/temp/'+ name + str(i) +'.jpg', _x1)

def plotheatmap(img, var, m=0, bw=False):
    import cv2
    import numpy as np
    _img = img.data.cpu().numpy()

    cols = 8
    rows = 8
    page = 0
    pages = 4
    _, _, _h, _w = _img.shape
    nh, nw = 255, 255
    if bw:
        imgcat = np.zeros((rows * nh, cols * nw))
    else:
        imgcat = np.zeros((rows * nh, cols * nw, 3))

    for i in range(_img.shape[1]):
        _row = i // cols % rows

        _x1 = _img[0, i, :, :]
        # m = np.median(_x1)
        # m = np.unique(_x1)
        if m > -100:
            _x1[_x1 < m] = 0

        _x1 = cv2.resize(_img[0, i, :, :], (nh, nw), interpolation=cv2.INTER_AREA)
        _x1 = cv2.normalize(_x1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        # cv2.imwrite('../../debug/demo/de_bug_2.jpg', _x1)
        _col = i % cols
        if bw:
            imgcat[_row * nh:(_row + 1) * nh, _col * nw:_col * nw + nw] = _x1
            imgcat[(_row + 1) * nh - 5:(_row + 1) * nh, _col * nw:_col * nw + nw] = np.ones((5, nw))
            imgcat[_row * nh:(_row + 1) * nh, _col * nw + nw - 5:_col * nw + nw] = np.ones((nh, 5))
        else:
            _x1 = cv2.applyColorMap(_x1, cv2.COLORMAP_JET)
            imgcat[_row * nh:(_row + 1) * nh, _col * nw:_col * nw + nw, :] = _x1
            imgcat[(_row + 1) * nh - 5:(_row + 1) * nh, _col * nw:_col * nw + nw, :] = np.ones((5, nw, 3))
            imgcat[_row * nh:(_row + 1) * nh, _col * nw + nw - 5:_col * nw + nw, :] = np.ones((nh, 5, 3))

        if _row == rows - 1 and i % cols == cols - 1:
            cv2.imwrite('/media/wattanapongsu/4T/temp/' + var + '_' + str(page) + ('.png' if bw else '.jpg'), imgcat)
            if bw:
                imgcat = np.zeros((rows * nh, cols * nw))
            else:
                imgcat = np.zeros((rows * nh, cols * nw, 3))
            page += 1
