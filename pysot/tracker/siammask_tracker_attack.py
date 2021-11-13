# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np

from pysot.core.config import cfg
from pysot.utils.bbox import cxy_wh_2_rect
from pysot.tracker.siamrpn_tracker import SiamRPNTracker
import os
from pysot.core.config import cfg
import torch
from pysot.utils.miscellaneous import get_subwindow_custom

torch.manual_seed(1999)
np.random.seed(1999)


class SiamMaskTrackerAttack(SiamRPNTracker):
    def __init__(self, model):
        super(SiamMaskTrackerAttack, self).__init__(model)
        assert hasattr(self.model, 'mask_head'), \
            "SiamMaskTrackerAttack must have mask_head"
        assert hasattr(self.model, 'refine_head'), \
            "SiamMaskTrackerAttack must have refine_head"

    def _crop_back(self, image, bbox, out_sz, padding=0):
        a = (out_sz[0] - 1) / bbox[2]
        b = (out_sz[1] - 1) / bbox[3]
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz[0], out_sz[1]),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=padding)
        return crop

    def _mask_post_processing(self, mask):
        target_mask = (mask > cfg.TRACK.MASK_THERSHOLD)
        target_mask = target_mask.astype(np.uint8)
        if cv2.__version__[-5] == '4':
            contours, _ = cv2.findContours(target_mask,
                                           cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_NONE)
        else:
            _, contours, _ = cv2.findContours(target_mask,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
        cnt_area = [cv2.contourArea(cnt) for cnt in contours]
        if len(contours) != 0 and np.max(cnt_area) > 100:
            contour = contours[np.argmax(cnt_area)]
            polygon = contour.reshape(-1, 2)
            prbox = cv2.boxPoints(cv2.minAreaRect(polygon))
            rbox_in_img = prbox
        else:  # empty mask
            location = cxy_wh_2_rect(self.center_pos, self.size)
            rbox_in_img = np.array([[location[0], location[1]],
                                    [location[0] + location[2], location[1]],
                                    [location[0] + location[2], location[1] + location[3]],
                                    [location[0], location[1] + location[3]]])
        return rbox_in_img


    def save(self, fname, tensor_img):
        cv2.imwrite(fname, cv2.UMat(tensor_img.data.cpu().numpy().squeeze().transpose([1, 2, 0])).get())


    def init(self, img, bbox, savedir, k=1, attacker=None):
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2, bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # self.channel_average = torch.mean(img, dim=(0, 1))
        self.channel_average = np.mean(img, axis=(0, 1))

        self.z_crop, box, pad = get_subwindow_custom(img, self.center_pos,
                                        cfg.TRACK.EXEMPLAR_SIZE,
                                        s_z, self.channel_average)

        h, w, _ = img.shape

        box[0] = box[0] - pad[0]
        box[1] = box[1] - pad[0]
        box[2] = box[2] - pad[2]
        box[3] = box[3] - pad[2]

        box[0] = 0 if box[0] < 0 else box[0]
        box[2] = 0 if box[2] < 0 else box[2]
        box[1] = h - 1 if box[1] > h else box[1]
        box[3] = w - 1 if box[3] > w else box[3]

        attacker.set_input(self.z_crop)
        with torch.no_grad():
            z_adv, z_delta = attacker.gen_noise(128, k)
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            self.save(os.path.join(savedir, 'adv.jpg'), z_adv)
            self.save(os.path.join(savedir, 'init.jpg'), self.z_crop)
            self.save(os.path.join(savedir, 'delta.jpg'), z_delta)

            mae = (z_adv - self.z_crop).abs().mean().data.cpu().numpy()

            text_file = open(os.path.join(savedir, 'MAE_'+str(mae)), "w")
            n = text_file.write(str(mae))

            text_file.close()
            self.model.template(z_adv)
            if k == -1:
                if len(self.model.zf) > 1:
                    self.model.zf[0] = torch.zeros_like(self.model.zf[0]).cuda()
                    self.model.zf[1] = torch.zeros_like(self.model.zf[1]).cuda()
                    self.model.zf[2] = torch.zeros_like(self.model.zf[2]).cuda()
                else:
                    self.model.zf = torch.zeros_like(self.model.zf).cuda()
        return s_z, box, pad, mae


    def track(self, img, attacker=None, epsilon=0, idx=0, iter=0, attack_search=False, debug=False):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        s_x = round(s_x)

        x_crop = self.get_subwindow(img,
                                    self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    s_x,
                                    self.channel_average)
        crop_box = [self.center_pos[0] - s_x / 2,
                    self.center_pos[1] - s_x / 2,
                    s_x,
                    s_x]

        with torch.no_grad():
            outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))
        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)

        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy,
                                                width, height, img.shape[:2])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]

        # processing mask
        pos = np.unravel_index(best_idx, (5, self.score_size, self.score_size))
        delta_x, delta_y = pos[2], pos[1]

        mask = self.model.mask_refine((delta_y, delta_x)).sigmoid().squeeze()
        out_size = cfg.TRACK.MASK_OUTPUT_SIZE
        mask = mask.view(out_size, out_size).cpu().data.numpy()

        s = crop_box[2] / cfg.TRACK.INSTANCE_SIZE
        base_size = cfg.TRACK.BASE_SIZE
        stride = cfg.ANCHOR.STRIDE
        sub_box = [crop_box[0] + (delta_x - base_size/2) * stride * s,
                   crop_box[1] + (delta_y - base_size/2) * stride * s,
                   s * cfg.TRACK.EXEMPLAR_SIZE,
                   s * cfg.TRACK.EXEMPLAR_SIZE]
        s = out_size / sub_box[2]

        im_h, im_w = img.shape[:2]
        back_box = [-sub_box[0] * s, -sub_box[1] * s, im_w*s, im_h*s]
        mask_in_img = self._crop_back(mask, back_box, (im_w, im_h))
        polygon = self._mask_post_processing(mask_in_img)
        polygon = polygon.flatten().tolist()
        return {
                'bbox': bbox,
                'best_score': best_score,
                'mask': mask_in_img,
                'polygon': polygon,
               }
