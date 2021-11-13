from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.neck import get_neck
from pysot.utils.data_utils import normalize
from pix2pix.models.base_model import BaseModel
from pix2pix.models import networks
from pysot.models.model_builder import ModelBuilder

torch.manual_seed(1999)

cls_thres = 0.7


class ModelAttacker(BaseModel):

    def set_input(self, input):
        pass

    def setConfig(self, cfg):
        self.cfg = cfg

    def optimize_parameters(self):
        pass

    def forward(self):
        pass

    def __init__(self, opt, tracker, GPUs=[0]):
        BaseModel.__init__(self, opt)

        self.netG = networks.define_G(3, 3, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type,
                                      opt.init_gain, GPUs)
        self.tracker = tracker
        self.opt = opt
        self.GPUs = GPUs

        if self.isTrain:
            # define loss functions
            self.criterionL2_noise = torch.nn.MSELoss(reduction='mean')
            self.criterionL2_encode = torch.nn.MSELoss()
            self.criterionL1 = torch.nn.L1Loss(reduction='mean')

            self.margin = -5
            self.cls_margin = -0.5
            self.reg_margin = -5

            self.l1 = self.opt.l1
            self.l2 = self.opt.l2
            self.weight_cls = self.opt.cls
            self.weight_reg = self.opt.reg
            self.curl_lr = opt.lr
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            # self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr)
            # self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr=opt.lr, momentum=0.9)
            #self.optimizers.append(self.optimizer_G)

    def learn(self, data, z_size=128, search_attack=False):

        if search_attack:
            z = data['search'].cuda()
        else:
            # z, x tensor, shape=(batch,3,127,127) and (batch,3,255,255) [0,255]
            z = data['template'].cuda()

        z_clean255 = z.cuda()
        z_clean1 = normalize(z_clean255)
        batch, ch, w, h = z_clean255.shape

        ze_clean = torch.zeros(batch, ch, w + 1, h + 1).cuda()
        _ze_clean = torch.zeros(batch, ch, z_size, z_size).cuda()
        ze_clean[:, :, 1:, 1:] = z_clean1

        if z_size != 128:
            _ze_clean = torch.nn.functional.interpolate(ze_clean, [z_size, z_size], mode='bilinear')
            _ze_noise = self.netG(_ze_clean).cuda()
            ze_noise = torch.nn.functional.interpolate(_ze_noise, [w + 1, h + 1], mode='bilinear')
        else:
            ze_noise = self.netG(ze_clean).cuda()

        ze_adv = ze_clean + ze_noise

        '''Then crop back to (127,127)'''
        z_adv1 = ze_adv[:, :, 1:, 1:]
        # z_adv255 = torch.clamp(z_adv1 * 127.5 + :q:q!127.5, min=0, max=255)
        z_adv255 = torch.clamp(255 * (z_adv1 * 0.5 + 0.5), min=0, max=255)

        z_noise255 = z_adv255 - z_clean255
        _z_noise255 = 10 * (z_noise255 - z_noise255.min(2)[0].min(2)[0].min(1)[0].view(z_noise255.shape[0], 1, 1, 1))
        zf_adv = self.tracker.forward_z(z_adv255)
        with torch.no_grad():
            zf_clean = self.tracker.forward_z(z_clean255)
        loss, loss_noise, mal_loss = self.backward_2l(z_noise255, zf_adv, zf_clean)

        '''backward pass'''
        # update G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        loss.backward()
        self.optimizer_G.step()  # update G's weights

        return {
            'loss': loss.data.item(),
            'loss_noise': loss_noise,
            'loss_mal': mal_loss,
            'z_adv': z_adv255.data,
            'z_noise': _z_noise255.data
        }

    def gen_noise(self, x, dim, k=1):

        if x.shape[2] == 127:
            _x = x.clone()
        else:
            bw, bh = (x.shape[2] + 1) // 2, (x.shape[3] + 1) // 2
            _x = x[:, :, bw - 64:bw + 64 - 1, bh - 64:bh + 64 - 1].clone()
        # pdb.set_trace()
        xnorm = normalize(_x)
        _, ch, w, h = _x.shape
        x_clean = torch.zeros(1, ch, w + 1, h + 1).cuda()
        x_clean[:, :, 1:, 1:] = xnorm

        if dim == x.shape[2] + 1:
            x_noise = k * self.netG(x_clean)
        else:
            _x_clean = torch.nn.functional.interpolate(x_clean, [dim, dim], mode='bilinear')
            _x_noise = k * self.netG(_x_clean)
            x_noise = torch.nn.functional.interpolate(_x_noise, [w + 1, h + 1], mode='bilinear')

        x_adv_norm = x_clean + x_noise
        _x = x_adv_norm[:, :, 1:, 1:]
        _x = torch.clamp(255 * (_x * 0.5 + 0.5), min=0, max=255)
        x_adv = x.clone()
        if x.shape[2] == 127:
            x_adv = _x
        else:
            bw, bh = (x.shape[2] + 1) // 2, (x.shape[3] + 1) // 2
            x_adv[:, :, bw - 64:bw + 64 - 1, bh - 64:bh + 64 - 1] = _x
        return x_adv

    def gen_noise_search(self, x, dim, k=1):

        # x.shape[2] == 255:
        _x = x.clone()

        # pdb.set_trace()
        xnorm = normalize(_x)
        _, ch, w, h = _x.shape
        x_clean = torch.zeros(1, ch, w + 1, h + 1).cuda()
        x_clean[:, :, 1:, 1:] = xnorm

        if dim == x.shape[2] + 1:
            x_noise = k * self.netG(x_clean)
        else:
            _x_clean = torch.nn.functional.interpolate(x_clean, [dim, dim], mode='bilinear')
            _x_noise = k * self.netG(_x_clean)
            x_noise = torch.nn.functional.interpolate(_x_noise, [w + 1, h + 1], mode='bilinear')

        x_adv_norm = x_clean + x_noise
        _x = x_adv_norm[:, :, 1:, 1:]
        _x = torch.clamp(255 * (_x * 0.5 + 0.5), min=0, max=255)

        return _x

    def backward_noise(self, z_noise255):
        batch, channel, _, _ = z_noise255.shape
        z_noise255 = z_noise255.view(batch, -1)
        loss_noise = (z_noise255*z_noise255).mean(1).mean()
        return loss_noise

    def plot3d(self, dz):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt
        import numpy as np
        import math

        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')
        dz = dz.data.cpu().numpy()
        n1 = int(math.sqrt(len(dz)))

        d1 = np.arange(n1)
        xpos = np.tile(d1, n1)
        ypos = np.repeat(d1, n1)
        zpos = np.zeros(n1 * n1)

        dx = np.ones(n1 * n1)
        dy = np.ones(n1 * n1)

        # print(dz)
        ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)  # , color='#0000FF'
        plt.show()

    def backward_enc(self, zf_adv, zf_clean):
        ch = 1
        if isinstance(zf_adv, list):
            # zf_adv.shape = [3, batch, 256, 7, 7]
            zf_adv_stack = torch.stack(zf_adv)
            zf_clean_stack = torch.stack(zf_clean)
            ch, batch, _, _, _ = zf_adv_stack.shape

        else:
            # zf_adv.shape = [batch, 256, 6, 6]
            batch, _, _, _ = zf_adv.shape
            zf_adv_stack = zf_adv
            zf_clean_stack = zf_clean

        # Replace clean feature's elements are less than 0.8 to 0
        zfa = zf_adv_stack.contiguous().view(ch * batch, -1)
        zfc = zf_clean_stack.contiguous().view(ch * batch, -1)

        zfcn = zfc / zfc.abs().max(dim=1)[0].contiguous().view(ch * batch, 1)
        pdb.set_trace()
        # self.plot3d(zfc[0].view(256, -1)[0])
        # self.plot3d(zfa[0].view(256, -1)[0])
        # self.plot3d(zfcn[0].view(256, -1)[0])
        pos_mal = zfcn.abs() > self.opt.droprate
        pos_fix = zfcn.abs() <= self.opt.droprate
        # num = ch * batch * zfc.shape[1]
        # print(100 * pos_mal.sum().data.cpu().numpy() / num, 100 * pos_fix.sum().data.cpu().numpy() / num, num)

        loss = torch.tensor(0, dtype=torch.float32).cuda()
        # pdb.set_trace()
        num_train = int(self.opt.nrpn * zfa.shape[0]/3)
        for i in range(num_train):
            _zf_fix = (zfa[i, pos_fix[i, :]] * zfa[i, pos_fix[i, :]]).mean()
            _zf_mal = (zfa[i, pos_mal[i, :]] * zfa[i, pos_mal[i, :]]).mean()
            # loss += torch.clamp(_zf_mal - _zf_fix, min=-0.05)
            # print(100 * pos_mal[i, :].sum() / zfa.shape[1])
            loss += _zf_mal - _zf_fix
            # print(loss.item())
        # print('----------------------------')
        return loss / num_train

    def backward_2l(self, z_noise255, zf_adv, zf_clean):

        loss_noise = self.backward_noise(z_noise255) * self.l1
        loss_noise_val = loss_noise.data.item()
        loss = loss_noise

        mal_loss = self.backward_enc(zf_adv, zf_clean) * self.l2

        mal_loss_val = mal_loss.data.item()
        loss += mal_loss

        return loss, loss_noise_val, mal_loss_val


class ModelTracker(ModelBuilder):

    def __init__(self):
        ModelBuilder.__init__(self)

    def forward_z(self, z):

        zf = self.backbone(z)

        if cfg.MASK.MASK:
            zf = zf[-1]

        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)

        return zf

    def forward(self, z, x):

        zf = self.backbone(z)
        with torch.no_grad():
            xf = self.backbone(x)

        if cfg.MASK.MASK:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            with torch.no_grad():
                xf = self.neck(xf)

        cls, loc = self.rpn_head(zf, xf)
        cls = cls.permute(1, 2, 3, 0).contiguous().view(2, -1)
        loc = loc.permute(1, 2, 3, 0).contiguous().view(4, -1)
        return zf, cls, loc
