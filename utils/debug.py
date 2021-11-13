import torch
import os
import pdb

from utils.myplot import *

zf = torch.load('/media/wattanapongsu/4T/temp/save/OTB100/reg_8e3_e13_k-1.0/Basketball/zf.pth')
zfo = zf['zfo']
zfa = zf['zfa']
x = (zfa - zfo).sum(dim=[3,4]).squeeze(dim=1)

plot((zfo - zfa).abs().sum(dim=[3,4]).squeeze(dim=1)/49)
plot(zfo.sum(dim=[3,4]).squeeze(dim=1))
pdb.set_trace()

