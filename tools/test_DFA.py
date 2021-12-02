# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import time
import math
import json
import random
import numpy as np
from tqdm import tqdm
import pdb
import cv2
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pysot.utils.log_helper import init_log, print_speed, add_file_handler
from pysot.utils.model_load import load_pretrain
from pysot.utils.misc import commit
from pysot.models.model_builder import ModelBuilder
from pysot.core.config import cfg
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.data_utils import *
from toolkit.utils.region import vot_overlap, vot_float2str
from pysot.utils.bbox import get_axis_aligned_bbox, get_axis_aligned_bbox_tensor
from toolkit.datasets import DatasetFactory
from pix2pix.options.test_options import TestOptions
from pysot.models.model_attack_template import ModelAttacker

logger = logging.getLogger('global')
parser = argparse.ArgumentParser(description='siamrpn++ template attacking')
parser.add_argument('--cfg', type=str, default='config.yaml',
                    help='configuration of tracking')
parser.add_argument('--dataset', type=str,
                    help='datasets')
parser.add_argument('--dataset_dir', type=str,
                    default='/4T/dataset',
                    help='dataset directory')
parser.add_argument('--saved_dir', default='', type=str,
                    help='save images and videos in this directory')
parser.add_argument('--fabricated_dir', default='', type=str,
                    help='save images and videos in this directory')
parser.add_argument('--snapshot', default='', type=str,
                    help='snapshot of models to eval')
parser.add_argument('--freq', default=20, type=int,
                    help='display frequency')
parser.add_argument('--video', default='', type=str,
                    help='eval one special video')
parser.add_argument('--seed', type=int, default=123456,
                    help='random seed')
parser.add_argument('--netG_pretrained', default='', type=str,
                            help='netG pretrained ')
parser.add_argument('--model_name', default='', type=str,
                            help='model name ')
parser.add_argument('--k', default=1, type=float,
                            help='amplified parameter in template')
parser.add_argument('--ks', default=1, type=float,
                            help='amplified parameter in searching')
parser.add_argument('--chk', default=1, type=int,
                            help='checkpoint number')
parser.add_argument('--export_video', action='store_true',
                            help='export video output')
parser.add_argument('--gpus', default=0, type=int,
                            help='number of gpus')
parser.add_argument('--search_attack', action='store_true',
                            help='attack search image')
parser.add_argument('--z_size', type=int, default=128,
                            help='interpolate template size')
parser.add_argument('--model_search', action='store_true',
                            help='generate noise in search style')

args = parser.parse_args()


def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def gt_bbox_adaptor(gt_bbox):
    if len(gt_bbox) == 4:
        gt_bbox = [gt_bbox[0], gt_bbox[1],
                   gt_bbox[0], gt_bbox[1] + gt_bbox[3] - 1,
                   gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1] + gt_bbox[3] - 1,
                   gt_bbox[0] + gt_bbox[2] - 1, gt_bbox[1]]
    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    return gt_bbox, gt_bbox_


def stoa_track(idx, frame_counter, img, gt_bbox, tracker, savedir, attacker=None):
    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    lost_number = 0
    mae = 0
    if idx == frame_counter:
        init_gt = gt_bbox_
        _, _, _, mae = tracker.init(img, gt_bbox_, savedir, args.k, attacker=attacker, z_size=args.z_size)
        # pred_bboxes.append(1)
        if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
            append = 1
        else:
            append = gt_bbox_

        overlap = 1

    elif idx > frame_counter:
        outputs = tracker.track(img, idx=idx, k=args.ks, export=args.export_video,
                                attacker=attacker, search_attack=args.search_attack,
                                z_size=args.z_size, model_search=args.model_search)
        mae = outputs['mae']
        # print('****************** state of the art tracking ******************')
        append = outputs['bbox']

        overlap = vot_overlap(outputs['bbox'], gt_bbox, (img.shape[1], img.shape[0]))
        if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
            if overlap > 0:
                # not lost
                lost = False
            else:
                # lost object
                append = 2
                frame_counter = idx + 5  # skip 5 frames
                lost_number = 1
                lost = True
        else:
            if overlap <= 0:
                lost_number = 1

    else:
        append = 0
        overlap = -1

    return append, lost_number, frame_counter, overlap, mae


def test(video, tracker, model_name, savedir, v_idx, num_video, attacker=None):

    # set writing video parameters
    height, width, channels = video[0][0].shape
    if not os.path.exists(os.path.join(args.saved_dir, args.dataset, model_name)):
        os.makedirs(os.path.join(args.saved_dir, args.dataset, model_name))
    if args.export_video:
        out = cv2.VideoWriter(os.path.join(args.saved_dir, args.dataset, model_name, video.name + '.avi'),
                              cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 15, (width, height))
    frame_counter = 0
    toc = 0
    pred_bboxes_adv = []
    IOUs = []
    adv_z = []
    lost = 0
    mae = 0
    maes = []

    pbar = tqdm(enumerate(video), position=0, leave=True)

    for idx, (img, gt_bbox) in pbar:
        pbar.set_postfix_str('frame %d: video %d/%d' % (idx, v_idx, num_video))
        gt_bbox, gt_bbox_ = gt_bbox_adaptor(gt_bbox)

        tic = cv2.getTickCount()
        pred_bbox, _lost, frame_counter, iou, _mae = \
            stoa_track(idx, frame_counter, img, gt_bbox, tracker, savedir, attacker)

        if args.search_attack:
            maes.append(_mae)
        if idx == 0:
            mae = _mae
        pred_bboxes_adv.append(pred_bbox)
        IOUs.append(iou)
        toc += cv2.getTickCount() - tic
        lost += _lost

        if idx > 0:
            if not isinstance(pred_bbox, int):
                bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 3)

        gtskip = False
        if len(gt_bbox_) == 4:
            for _i in range(0, 4):
                if math.isnan(gt_bbox_[_i]):
                    gtskip = True

        if not gtskip:
            __gt_bbox = list(map(int, gt_bbox_))
            cv2.rectangle(img, (__gt_bbox[0], __gt_bbox[1]),
                          (__gt_bbox[0] + __gt_bbox[2], __gt_bbox[1] + __gt_bbox[3]), (255, 0, 0), 3)

        # cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)
        # cv2.putText(img, str(idx), (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 2)

        cv2.putText(img, str(idx), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        if args.export_video:
            out.write(img)

    # save results
    if args.dataset not in ['VOT2016', 'VOT2018', 'VOT2019']:
        model_path = os.path.join('results', args.dataset, model_name)
        if not os.path.isdir(model_path):
            os.makedirs(model_path)

        result_path = os.path.join(model_path, '{}.txt'.format(video.name))
        with open(result_path, 'w') as f:
            for x in pred_bboxes_adv:
                f.write(','.join([str(i) for i in x]) + '\n')

        iou_path = os.path.join(model_path, '{}_iou.txt'.format(video.name))
    else:
        video_path = os.path.join('results', args.dataset, model_name,
                                  'baseline', video.name)
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, '{}_001.txt'.format(video.name))

        with open(result_path, 'w') as f:
            for x in pred_bboxes_adv:
                if isinstance(x, int):
                    f.write("{:d}\n".format(x))
                else:
                    f.write(','.join([vot_float2str("%.4f", i) for i in x]) + '\n')

        iou_path = os.path.join(video_path, '{}_iou.txt'.format(video.name))

    # write IOU
    with open(iou_path, 'w') as f:
        for x in IOUs:
            f.write(str(x) + '\n')

    return mae, maes

def main():
    # load config
    cfg.merge_from_file(args.cfg)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    dataset_root = os.path.join(args.dataset_dir, args.dataset)

    # create model
    track_model = ModelBuilder()
    track_model = load_pretrain(track_model, args.snapshot).cuda().eval()
    # build tracker
    tracker = build_tracker(track_model)

    opt = TestOptions().parse(load=False)
    opt.eval = True
    if opt.eval:
        print('evaluate...')

    attacker = ModelAttacker(opt, tracker)
    attacker.netG.cuda().eval()

    # load netG pretrained
    netG_pretrained_path = os.path.join(args.saved_dir, 'checkpoint', args.model_name,
                                        opt.netG_pretrained)
    print('load netG pretrained ... ', netG_pretrained_path)
    state = torch.load(netG_pretrained_path)
    attacker.netG.load_state_dict(state['netG'])

    # create dataset
    if args.dataset == 'LaSOT' or args.dataset == 'UAV123' or args.dataset == 'GOT-10k':
        dataset = DatasetFactory.create_dataset(name=args.dataset,
                                                dataset_root=dataset_root,
                                                load_img=False)
    else:
        dataset = DatasetFactory.create_dataset(name=args.dataset,
                                            dataset_root=dataset_root,
                                            load_img=False,
                                            dataset_toolkit='oneshot',
                                            config=cfg)

    mae = []

    model_name = args.model_name
    extra = ''
    if args.search_attack:
        if args.k == 0:
            extra = '_s'
        else:
            extra = '_ts'
    else:
        extra = '_t'
    model_name += extra+'_e' + str(args.chk) + '_k' + str(args.k)
    # if args.k != 1:
    #     model_name += '_k' + str(args.k)

    savedir = os.path.join(args.saved_dir, args.dataset, model_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019', 'OTB100', 'LaSOT', 'UAV123', 'GOT-10k']:

        time_start = cv2.getTickCount()

        # restart tracking
        for v_idx, video in enumerate(dataset):

            # for multiple GPUs
            if args.gpus > 0:
                if args.dataset in ['VOT2016', 'VOT2018', 'VOT2019']:
                    video_path = os.path.join('results', args.dataset, model_name, 'baseline', video.name,
                                              '{}_001.txt'.format(video.name))
                else:
                    video_path = os.path.join('results', args.dataset, model_name, '{}.txt'.format(video.name))

                if os.path.exists(video_path):
                    continue

                if os.path.exists(os.path.join(savedir, 'occupy_' + video.name)):
                    continue
                else:
                    file = open(os.path.join(savedir, 'occupy_' + video.name), 'w+')

            # img_names = [x.replace(args.dataset_dir, args.fabricated_dir) for x in video.img_names]
            # fabricated_video(img_names, video)

            # myDataset =
            video_saved_dir = os.path.join(args.saved_dir, args.dataset, model_name, video.name)
            if args.video != '':
                # test one special video
                if video.name != args.video:
                    continue
                else:
                    if not os.path.exists(video_saved_dir):
                        os.makedirs(video_saved_dir)
            else:
                if not os.path.exists(video_saved_dir):
                    os.makedirs(video_saved_dir)

            # elif v_idx < args.video_idx and args.debug:
            #     continue

            ##########################################
            # # #  for state of the art tracking # # #
            ##########################################
            img_names = [x for x in video.img_names]

            _mae, maes = test(video, tracker, model_name, video_saved_dir, v_idx, len(dataset), attacker)

            if args.search_attack:
                maes_path = os.path.join(savedir, video.name + '.txt')

                with open(maes_path, 'w') as f:
                    for x in maes:
                        f.write(str(x) + '\n')

            mae.append([video.name, str(_mae)])

            # for multiple GPUs
            if args.gpus > 0:
                os.remove(os.path.join(savedir, 'occupy_' + video.name))

        time_usage = (cv2.getTickCount() - time_start) / cv2.getTickFrequency()

        print('time usage: ' + str(time_usage))

        # write MAE for all video sequences
        if args.gpus == 0:
            mae_path = os.path.join(savedir, model_name+'.txt')

            with open(mae_path, 'w') as f:
                for x in mae:
                    f.write(','.join([i for i in x]) + '\n')


if __name__ == '__main__':
    seed_torch(args.seed)
    main()