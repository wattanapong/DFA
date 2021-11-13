# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import cv2
import torch, pdb, os

from pysot.core.config import cfg
from pysot.utils.anchor import Anchors
from pysot.tracker.base_tracker import SiameseTracker
from pysot.datasets.anchor_target import AnchorTarget
from pysot.utils.bbox import get_min_max_bbox, center2corner, Center, get_axis_aligned_bbox
from pysot.utils.miscellaneous import get_subwindow_custom
from pysot.utils.heatmap import plotheatmap, maprpn, maprpn_id
from pysot.utils.data_utils import normalize
torch.manual_seed(1999)
np.random.seed(1999)


class SiamRPNAttackFeature(SiameseTracker):
    def __init__(self, model):
        super(SiamRPNAttackFeature, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
                          cfg.ANCHOR.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        self.anchor_num = len(cfg.ANCHOR.RATIOS) * len(cfg.ANCHOR.SCALES)
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.window = np.tile(window.flatten(), self.anchor_num)
        self.anchors = self.generate_anchor(self.score_size)
        self.model = model
        self.model.eval()
        self.zf = None
        self.zfa = None
        self.shift = None
        self.anchor_target = AnchorTarget()

    def generate_transition(self, shift, num):
        self.shift = torch.from_numpy(shift * np.random.rand(2, num) - shift // 2).float()

    def generate_anchor(self, score_size):
        anchors = Anchors(cfg.ANCHOR.STRIDE,
                          cfg.ANCHOR.RATIOS,
                          cfg.ANCHOR.SCALES)
        anchor = anchors.anchors
        x1, y1, x2, y2 = anchor[:, 0], anchor[:, 1], anchor[:, 2], anchor[:, 3]
        anchor = np.stack([(x1 + x2) * 0.5, (y1 + y2) * 0.5, x2 - x1, y2 - y1], 1)
        total_stride = anchors.stride
        anchor_num = anchor.shape[0]
        anchor = np.tile(anchor, score_size * score_size).reshape((-1, 4))
        ori = - (score_size // 2) * total_stride
        xx, yy = np.meshgrid([ori + total_stride * dx for dx in range(score_size)],
                             [ori + total_stride * dy for dy in range(score_size)])
        xx, yy = np.tile(xx.flatten(), (anchor_num, 1)).flatten(), \
                 np.tile(yy.flatten(), (anchor_num, 1)).flatten()
        anchor[:, 0], anchor[:, 1] = xx.astype(np.float32), yy.astype(np.float32)
        return anchor

    def _convert_bboxes(self, delta, anchor):
        batch = delta.shape[0]
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1, batch)

        anchor = torch.from_numpy(anchor).cuda()

        anchor = anchor.contiguous().view(-1, 1).repeat(1, batch).view(-1, 4, batch)
        delta[0, :, :] = delta[0, :, :] * anchor[:, 2, :] + anchor[:, 0, :]
        delta[1, :, :] = delta[1, :, :] * anchor[:, 3, :] + anchor[:, 1, :]

        delta[2, :, :] = torch.exp(delta[2, :, :]) * anchor[:, 2, :]
        delta[3, :, :] = torch.exp(delta[3, :, :]) * anchor[:, 3, :]

        return delta

    def _convert_scores(self, score):
        batch = score.shape[0]
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1, batch).permute(2, 1, 0)
        score_softmax = F.softmax(score, dim=2)[:, :, 1]

        return score_softmax

    def _convert_bbox(self, delta, anchor):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)

        anchor = torch.from_numpy(anchor).cuda()

        delta[0, :] = delta[0, :] * anchor[:, 2] + anchor[:, 0]
        delta[1, :] = delta[1, :] * anchor[:, 3] + anchor[:, 1]

        delta[2, :] = torch.exp(delta[2, :]) * anchor[:, 2]
        delta[3, :] = torch.exp(delta[3, :]) * anchor[:, 3]

        return delta

    def _convert_score(self, score):
        score = score.permute(1, 2, 3, 0).contiguous().view(2, -1).permute(1, 0)
        score_softmax = F.softmax(score, dim=1)[:, 1]
        return score_softmax

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[-2:]
        cx, cy = imw // 2, imh // 2

        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape

        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z

        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def crop(self, img, bbox=None, im_name=None):
        # calculate channel average
        self.channel_average = np.mean(img, axis=(0, 1))

        # [x, y, w, h] to [cx, cy, w, h]
        bbox = [bbox[0] + bbox[2] // 2, bbox[1] + bbox[3] // 2, bbox[2], bbox[3]]
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        sz = round(s_x)
        # s_x = sz * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        limit_size = cfg.TRACK.INSTANCE_SIZE

        # get crop
        # box = {top, bottom, left, right}
        # pad = {pad top, pad bottom, pad left, pad pad right}
        h, w, _ = img.shape
        _crop, box, pad = get_subwindow_custom(img, self.center_pos, limit_size, sz, self.channel_average)

        box[0] = box[0] - pad[0]
        box[1] = box[1] - pad[0]
        box[2] = box[2] - pad[2]
        box[3] = box[3] - pad[2]

        box[0] = 0 if box[0] < 0 else box[0]
        box[2] = 0 if box[2] < 0 else box[2]
        box[1] = h - 1 if box[1] > h else box[1]
        box[3] = w - 1 if box[3] > w else box[3]

        return _crop, sz, box, pad

    def save(self, fname, tensor_img):
        cv2.imwrite(fname, cv2.UMat(tensor_img.data.cpu().numpy().squeeze().transpose([1, 2, 0])).get())

    def init(self, img, bbox, savedir, k=1, attacker=None, z_size=128):
        self.savedir = savedir
        # img = torch.from_numpy(img).type(torch.FloatTensor)

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

        mae = 0
        with torch.no_grad():
            if attacker is not None and k != -99:
                if k != 0:
                    if k == -1:
                        _k = 1
                    else:
                        _k = k

                    z_adv = attacker.gen_noise(self.z_crop, z_size, _k)

                    z_delta = z_adv - self.z_crop
                    if not os.path.exists(savedir):
                        os.makedirs(savedir)

                    self.save(os.path.join(savedir, 'adv.jpg'), z_adv)
                    self.save(os.path.join(savedir, 'init.jpg'), self.z_crop)

                    # z_delta_min = z_delta.min(2)[0].min(2)[0].view(1, 3, 1, 1)
                    mae = z_delta.abs().mean().data.cpu().numpy()
                    zd = torch.clamp(z_delta.abs(), min=0, max=255)
                    self.save(os.path.join(savedir, 'delta.jpg'), zd)
                    self.save(os.path.join(savedir, 'delta10x.jpg'), 10 * zd)
                    pdb.set_trace()
                    text_file = open(os.path.join(savedir, 'MAE_'+str(mae)), "w")
                    text_file.write(str(mae)+'\n')
                    # text_file.write(str(l2norm) + '\n')
                    text_file.close()
                    self.model.template(z_adv)

            if attacker is None or k == 0:
                self.model.template(self.z_crop)

            if k == -99:
                self.model.template(self.z_crop)

                ch = 1
                if isinstance(self.model.zf, list):
                    # zf_adv.shape = [3, batch, 256, 7, 7]
                    zf_adv_stack = torch.stack(self.model.zf)
                    ch, batch, dim, w, h = zf_adv_stack.shape

                else:
                    # zf_adv.shape = [batch, 256, 6, 6]
                    batch, dim, w, h = self.model.zf.shape
                    zf_adv_stack = self.model.zf

                zfa = zf_adv_stack.contiguous().view(ch * batch* dim, -1)

                zfan = zfa.abs() / zfa.abs().max(dim=1)[0].contiguous().view(ch * batch * dim, -1)

                zfan_mean = zfan.mean(dim=1).contiguous().view(ch * batch * dim, 1)

                pos_mal = zfan >= zfan_mean

                # pdb.set_trace()

                zfa[pos_mal] = 0

                if len(self.model.zf) == 1:
                    zfa_stack = zfa.contiguous().view(batch, dim, w, h)
                    self.model.zf = zfa_stack
                else:
                    zfa_stack = zfa.contiguous().view(ch, batch, dim, w, h)
                    for i, zf in enumerate(self.model.zf):
                        self.model.zf[i] = zfa_stack[i]

        return s_z, box, pad, mae

    def track(self, img, attacker=None, epsilon=0, idx=0, k=1, export=False,
              iter=0, debug=False, search_attack=False, z_size=128, model_search=False):

        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)

        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop, _, _ = get_subwindow_custom(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)
        mae = 0
        with torch.no_grad():

            if search_attack and attacker is not None:
                if model_search:
                    x_adv = attacker.gen_noise_search(x_crop, z_size, k=k)
                else:
                    # time_start = cv2.getTickCount()
                    x_adv = attacker.gen_noise(x_crop, z_size, k=k)
                    # time_usage = (cv2.getTickCount() - time_start) / cv2.getTickFrequency()
                    # pdb.set_trace()
                    # print('{:.4f}'.format(time_usage))
                x_delta = x_adv - x_crop
                mae = x_delta.abs().mean().data.cpu().numpy()
                if export:
                    self.save(os.path.join(self.savedir, 'x' + str(idx).zfill(3) + '.jpg'), x_crop)
                    self.save(os.path.join(self.savedir, 'x_adv'+str(idx).zfill(3)+'.jpg'), x_adv)

                    xd = x_delta.abs() # torch.clamp(x_delta.abs(), min=0, max=255)
                    # xd = torch.clamp(x_delta - x_delta.min(), min=0, max=255)
                    self.save(os.path.join(self.savedir, 'delta_x10_'+str(idx).zfill(3)+'.jpg'), 10 * xd)
                outputs = self.model.track(x_adv)
                crop = x_adv
            else:
                outputs = self.model.track(x_crop)
                crop = x_crop

        score_softmax = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.anchors)

        def change(r):
            return torch.max(r, 1./r)

        def sz(w, h):
            if not torch.is_tensor(w):
                w = torch.tensor(w).type(torch.float)
            if not torch.is_tensor(h):
                h = torch.tensor(h).type(torch.float)
            pad = (w + h) * 0.5
            return torch.sqrt((w + pad) * (h + pad))

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) / (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change(torch.tensor(self.size[0]/self.size[1]).type(torch.float) / (pred_bbox[2, :]/pred_bbox[3, :]))

        penalty = torch.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K).cuda()
        pscore_softmax = penalty * score_softmax

        # window penalty
        pscore_softmax = pscore_softmax * torch.tensor(1 - cfg.TRACK.WINDOW_INFLUENCE, dtype=torch.float32).cuda() + \
        torch.tensor(self.window * cfg.TRACK.WINDOW_INFLUENCE, dtype=torch.float32).cuda()

        _, sort_idx = torch.sort(-pscore_softmax)

        # if attacker is not None:
        #     pdb.set_trace()

        best_idx = sort_idx[0]
        bbox = pred_bbox[:, best_idx].data.cpu().numpy() / scale_z

        best_score = score_softmax[best_idx]
        lr = (penalty[best_idx] * best_score * cfg.TRACK.LR).data.cpu().numpy()

        cx = bbox[0] + self.center_pos[0]
        cy = bbox[1] + self.center_pos[1]

        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width, height, img.shape[:2])

        # update state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        return {
            'bbox': bbox,
            'best_score': score_softmax[sort_idx[0]],
            'target_score': score_softmax[sort_idx[45 - 1]],
            'center_pos': np.array([cx, cy]),
            'size': np.array([width, height]),
            'mae': mae,
            'crop': crop
        }