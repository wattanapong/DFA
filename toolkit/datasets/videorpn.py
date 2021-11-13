import os
import cv2
import re
import numpy as np
import json

from glob import glob
from pysot.datasets.anchor_target import AnchorTarget
from pysot.utils.bbox import get_min_max_bbox, center2corner, Center, get_axis_aligned_bbox
from pysot.datasets.augmentation import Augmentation

class Video(object):
    def __init__(self, name, root, video_dir, init_rect, img_names,
            gt_rect, attr, load_img=False, config=None):
        self.name = name
        self.video_dir = video_dir
        self.init_rect = init_rect
        self.gt_traj = gt_rect
        self.attr = attr
        self.pred_trajs = {}

        dataset = root.split('/')[-1]
        if dataset == 'VOT2018' or dataset == 'VOT2016' or dataset == 'VOT2019':
            self.img_names = [os.path.join(root, x.replace('color/','')) for x in img_names]
        elif dataset == 'OTB100':
            if video_dir == 'Jogging-1' or 'Skating2-1' in video_dir:
                self.img_names = [os.path.join(root, x.replace('-1', '')) for x in img_names]
            elif video_dir == 'Jogging-2' or 'Skating2-2' in video_dir:
                self.img_names = [os.path.join(root, x.replace('-2', '')) for x in img_names]
            else:
                self.img_names = [os.path.join(root, x) for x in img_names]

        self.imgs = None
        self.config = config
        self.size = None
        self.center_pos = None

        self.template_aug = Augmentation(
            config.DATASET.TEMPLATE.SHIFT,
            config.DATASET.TEMPLATE.SCALE,
            config.DATASET.TEMPLATE.BLUR,
            config.DATASET.TEMPLATE.FLIP,
            config.DATASET.TEMPLATE.COLOR
        )
        self.search_aug = Augmentation(
            config.DATASET.SEARCH.SHIFT,
            config.DATASET.SEARCH.SCALE,
            config.DATASET.SEARCH.BLUR,
            config.DATASET.SEARCH.FLIP,
            config.DATASET.SEARCH.COLOR
        )

        # create anchor target
        self.anchor_target = AnchorTarget()

        if load_img:
            self.imgs = [cv2.imread(x) for x in self.img_names]
            self.width = self.imgs[0].shape[1]
            self.height = self.imgs[0].shape[0]
        else:
            img = cv2.imread(self.img_names[0])
            assert img is not None, self.img_names[0]
            self.width = img.shape[1]
            self.height = img.shape[0]

    def load_tracker(self, path, tracker_names=None, store=True):
        """
        Args:
            path(str): path to result
            tracker_name(list): name of tracker
        """
        if not tracker_names:
            tracker_names = [x.split('/')[-1] for x in glob(path)
                    if os.path.isdir(x)]
        if isinstance(tracker_names, str):
            tracker_names = [tracker_names]
        for name in tracker_names:
            traj_file = os.path.join(path, name, self.name+'.txt')
            if os.path.exists(traj_file):
                with open(traj_file, 'r') as f :
                    pred_traj = [list(map(float, x.strip().split(',')))
                            for x in f.readlines()]
                if len(pred_traj) != len(self.gt_traj):
                    print(name, len(pred_traj), len(self.gt_traj), self.name)
                if store:
                    self.pred_trajs[name] = pred_traj
                else:
                    return pred_traj
            else:
                print(traj_file)
        self.tracker_names = list(self.pred_trajs.keys())

    def load_img(self):
        if self.imgs is None:
            self.imgs = [cv2.imread(x) for x in self.img_names]
            self.width = self.imgs[0].shape[1]
            self.height = self.imgs[0].shape[0]

    def free_img(self):
        self.imgs = None

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        cx, cy = imw // 2, imh // 2

        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape

        context_amount = 0.5
        exemplar_size = self.config.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z

        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if self.imgs is None:
            img = cv2.imread(self.img_names[idx])
        else:
            img = self.imgs[idx]

        return img, self.gt_traj[idx]

    def perturb(self, bbox, sz):
        # cx, cy, w, h = get_axis_aligned_bbox(np.array(bbox))

        cx = (bbox.x1 + bbox.x2)/2
        cy = (bbox.y1 + bbox.y2)/2
        w = np.abs(bbox.x1 - bbox.x2)
        h = np.abs(bbox.y1 - bbox.y2)

        # w = np.abs(sz - w)/2
        # h = np.abs(sz - h)/2

        # cx = np.abs(sz - cx)
        # cy = np.abs(sz - cy)
        #
        # rx, ry = np.random.random(size=2)
        #
        # if sz/4 < cx < 3*sz/4 and rx > 0.5:
        #     cx = sz - cx
        # if sz/4 < cy < 3*sz/4 and ry > 0.5:
        #     cy = sz - cy

        # bbox = [cx-w/2, cy-y/2, cx-w/2, cy+y/2, cx+w/2, cy-y/2, cx+w/2, cy+y/2]

        # bbox = np.array([cx - w, cy - h, w//2, h//2])
        bbox = np.array([cx, cy, w, h])
        return center2corner(bbox)

    def get_subwindow_custom(self, im, pos, model_sz, original_sz, avg_chans):
        """
        args:
            im: bgr based image
            pos: center position
            model_sz: exemplar size
            s_z: original size
            avg_chans: channel average
        """
        if isinstance(pos, float):
            pos = [pos, pos]
        sz = original_sz
        im_sz = im.shape
        c = (original_sz + 1) / 2
        # context_xmin = round(pos[0] - c) # py2 and py3 round
        context_xmin = np.floor(pos[0] - c + 0.5)
        context_xmax = context_xmin + sz - 1
        # context_ymin = round(pos[1] - c)
        context_ymin = np.floor(pos[1] - c + 0.5)
        context_ymax = context_ymin + sz - 1
        left_pad = int(max(0., -context_xmin))
        top_pad = int(max(0., -context_ymin))
        right_pad = int(max(0., context_xmax - im_sz[1] + 1))
        bottom_pad = int(max(0., context_ymax - im_sz[0] + 1))

        context_xmin = context_xmin + left_pad
        context_xmax = context_xmax + left_pad
        context_ymin = context_ymin + top_pad
        context_ymax = context_ymax + top_pad

        r, c, k = im.shape
        if any([top_pad, bottom_pad, left_pad, right_pad]):
            size = (r + top_pad + bottom_pad, c + left_pad + right_pad, k)
            te_im = np.zeros(size, np.uint8)
            te_im[top_pad:top_pad + r, left_pad:left_pad + c, :] = im
            if top_pad:
                te_im[0:top_pad, left_pad:left_pad + c, :] = avg_chans
            if bottom_pad:
                te_im[r + top_pad:, left_pad:left_pad + c, :] = avg_chans
            if left_pad:
                te_im[:, 0:left_pad, :] = avg_chans
            if right_pad:
                te_im[:, c + left_pad:, :] = avg_chans
            im_patch = te_im[int(context_ymin):int(context_ymax + 1),
                             int(context_xmin):int(context_xmax + 1), :]
        else:
            im_patch = im[int(context_ymin):int(context_ymax + 1),
                          int(context_xmin):int(context_xmax + 1), :]

        if not np.array_equal(model_sz, original_sz):
            im_patch = cv2.resize(im_patch, (model_sz, model_sz))
        # im_patch = im_patch[np.newaxis, :, :, :]
        im_patch = im_patch.astype(np.int)

        return im_patch, [int(context_ymin), int(context_ymax), int(context_xmin), int(context_xmax)], \
            [top_pad, bottom_pad, left_pad, right_pad]

    def crop(self, img, bbox=None, is_template=True, im_name=None):
        # calculate channel average
        channel_average = np.mean(img, axis=(0, 1))

        # calculate z crop size
        if is_template:
            self.size = np.array([bbox[2], bbox[3]])

        w_z = self.size[0] + self.config.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + self.config.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        if is_template:
            sz = s_z
            limit_size = self.config.TRACK.EXEMPLAR_SIZE
            self.center_pos = np.array([bbox[0], bbox[1]])
        else:
            s_x = s_z * (self.config.TRACK.INSTANCE_SIZE / self.config.TRACK.EXEMPLAR_SIZE)
            sz = round(s_x)
            limit_size = self.config.TRACK.INSTANCE_SIZE

        h, w, _ = img.shape
        _crop, box, pad = self.get_subwindow_custom(img, self.center_pos, limit_size, sz, channel_average)

        box[0] = box[0] - pad[0]
        box[1] = box[1] - pad[0]
        box[2] = box[2] - pad[2]
        box[3] = box[3] - pad[2]

        box[0] = 0 if box[0] < 0 else box[0]
        box[2] = 0 if box[2] < 0 else box[2]
        box[1] = h-1 if box[1] > h else box[1]
        box[3] = w-1 if box[3] > w else box[3]

        return _crop, sz, box, pad

    def d__iter__(self):
        for i in range(len(self.img_names)):

            if self.imgs is not None:
                img = self.imgs[i]
            else:
                img = cv2.imread(self.img_names[i])

            # , cls, delta, delta_weight, bbox
            yield img, self.gt_traj[i]

    def __iter__(self):
        for i in range(len(self.img_names)):

            if self.imgs is not None:
                img = self.imgs[i]
            else:
                img = cv2.imread(self.img_names[i])

            # gray = self.config.DATASET.GRAY and self.config.DATASET.GRAY > np.random.random()
            gray = False

            # get Corner
            # gt_traj = {x1,y1,x2,y2,x3,y3,x4,y4}
            # bbox = {cx, cy, w, h}
            bbox = get_min_max_bbox(np.asarray(self.gt_traj[i], dtype=np.float32))

            # box = {top, bottom, left, right}
            z, szz, box, padz = self.crop(img, bbox=bbox, is_template=True, im_name='search')
            z = np.array(z.astype(np.uint8))
            h = box[1] - box[0]
            w = box[3] - box[2]
            _bbox = self._get_bbox(z, [w, h])

            # cv2.rectangle(nimg, (int(_bbox.x1), int(_bbox.y1)), (int(_bbox.x2), int(_bbox.y2)), (0, 0, 0), 3)
            # cv2.imwrite(os.path.join('/media/wattanapongsu/3T/temp/save', self.name, 'z.'+str(i).zfill(7)+'.jpg'), z)

            # exemplar, bbox_s = self.template_aug(nimg, _bbox, self.config.TRAIN.EXEMPLAR_SIZE, gray=gray)
            # nimg = cv2.UMat(nimg).get()
            # cv2.rectangle(exemplar, (int(bbox_s.x1), int(bbox_s.y1)), (int(bbox_s.x2), int(bbox_s.y2)), (0, 0, 0), 3)
            # cv2.imwrite(os.path.join('/media/wattanapongsu/3T/temp/save/bag/z.' + str(i) + '.2.jpg'), exemplar)

            x, szx, boxx, padx = self.crop(img, bbox=bbox, is_template=False, im_name='search')
            x = np.array(x.astype(np.uint8))
            h = boxx[1] - boxx[0]
            w = boxx[3] - boxx[2]

            # _bbox size is under 255x255
            _bbox = self._get_bbox(x, [w, h])

            # cv2.rectangle(nimg, (int(_bbox.x1), int(_bbox.y1)), (int(_bbox.x2), int(_bbox.y2)), (0, 0, 0), 3)
            # cv2.imwrite(os.path.join('/media/wattanapongsu/3T/temp/save', self.name, 'x.'+str(i).zfill(7)+'.jpg'), x)

            # search, bbox_s = self.search_aug(nimg, _bbox, self.config.TRAIN.SEARCH_SIZE, gray=gray)
            # cv2.rectangle(search, (int(bbox_s.x1), int(bbox_s.y1)), (int(bbox_s.x2), int(bbox_s.y2)), (0, 0, 0), 3)
            # cv2.imwrite(os.path.join('/media/wattanapongsu/3T/temp/save/bag/x.' + str(i) + '.2.jpg'), search)

            # get labels
            # bbox_perturb and _bbox are corner .x1, .y1, .x2, .y2
            bbox_perturb = self.perturb(_bbox, self.config.TRAIN.SEARCH_SIZE)
            cls_s, delta_s, delta_weight_s, overlap_s = self.anchor_target(
                bbox_perturb, self.config.TRAIN.OUTPUT_SIZE)

            import torch
            x = x.transpose(2, 0, 1)
            x = x[np.newaxis, :, :, :]
            x = x.astype(np.float32)
            x = torch.from_numpy(x)

            z = z.transpose(2, 0, 1)
            z = z[np.newaxis, :, :, :]
            z = z.astype(np.float32)
            z = torch.from_numpy(z)

            # , cls, delta, delta_weight, bbox
            yield img, self.gt_traj[i], z, x, szx, boxx, padx, cls_s, delta_s, delta_weight_s, overlap_s, _bbox, bbox_perturb

    def draw_box(self, roi, img, linewidth, color, name=None):
        """
            roi: rectangle or polygon
            img: numpy array img
            linewith: line width of the bbox
        """
        if len(roi) > 6 and len(roi) % 2 == 0:
            pts = np.array(roi, np.int32).reshape(-1, 1, 2)
            color = tuple(map(int, color))
            img = cv2.polylines(img, [pts], True, color, linewidth)
            pt = (pts[0, 0, 0], pts[0, 0, 1]-5)
            if name:
                img = cv2.putText(img, name, pt, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1)
        elif len(roi) == 4:
            if not np.isnan(roi[0]):
                roi = list(map(int, roi))
                color = tuple(map(int, color))
                img = cv2.rectangle(img, (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]),
                         color, linewidth)
                if name:
                    img = cv2.putText(img, name, (roi[0], roi[1]-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 1)
        return img

    def show(self, pred_trajs={}, linewidth=2, show_name=False):
        """
            pred_trajs: dict of pred_traj, {'tracker_name': list of traj}
                        pred_traj should contain polygon or rectangle(x, y, width, height)
            linewith: line width of the bbox
        """
        assert self.imgs is not None
        video = []
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        colors = {}
        if len(pred_trajs) == 0 and len(self.pred_trajs) > 0:
            pred_trajs = self.pred_trajs
        for i, (roi, img) in enumerate(zip(self.gt_traj,
                self.imgs[self.start_frame:self.end_frame+1])):
            img = img.copy()
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = self.draw_box(roi, img, linewidth, (0, 255, 0),
                    'gt' if show_name else None)
            for name, trajs in pred_trajs.items():
                if name not in colors:
                    color = tuple(np.random.randint(0, 256, 3))
                    colors[name] = color
                else:
                    color = colors[name]
                img = self.draw_box(trajs[0][i], img, linewidth, color,
                        name if show_name else None)
            cv2.putText(img, str(i+self.start_frame), (5, 20),
                    cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 0), 2)
            cv2.imshow(self.name, img)
            cv2.waitKey(40)
            video.append(img.copy())
        return video
