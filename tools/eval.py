from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
from toolkit.datasets import OTBDataset, UAVDataset, LaSOTDataset, \
        VOTDataset, NFSDataset, VOTLTDataset, GOT10kDataset
from toolkit.evaluation import OPEBenchmark, AccuracyRobustnessBenchmark, \
        EAOBenchmark, F1Benchmark
from pysot.core.config import cfg
from toolkit.visualization import draw_success_precision, draw_eao, draw_f1

parser = argparse.ArgumentParser(description='tracking evaluation')
parser.add_argument('--tracker_path', '-p', type=str,
                    help='tracker result path')
parser.add_argument('--dataset_path', type=str,
                    help='dataset path')
parser.add_argument('--dataset', '-d', type=str,
                    help='dataset name')
parser.add_argument('--num', '-n', default=1, type=int,
                    help='number of thread to eval')
parser.add_argument('--tracker_prefix', '-t', default='',
                    type=str, help='tracker name')
parser.add_argument('--show_video_level', '-s', dest='show_video_level',
                    action='store_true')
parser.add_argument('--vis', dest='vis', action='store_true')
parser.set_defaults(show_video_level=False)
parser.add_argument('--config', default='', type=str,
        help='config file')

args = parser.parse_args()


def save_vot_metrics(tracker_path, metrics, ar_result, tags):
    import csv
    metrics = dict(metrics)
    with open(tracker_path + 'metrics.csv', 'w') as f:

        for tracker_name in ar_result.keys():
            f.write('\t%s' % tracker_name)
        f.write("\n")

        for tag in tags:
            f.write("%s\t" % tag)

            for tracker_name in ar_result.keys():
                f.write("%.3f\t" % metrics[tracker_name][tag])
            f.write("\n")


def save_metrics(tracker_path, metrics, success_ret):
    import csv

    with open(tracker_path + 'metrics.csv', 'w') as f:

        f.write("Success\n\n")
        for tracker_name in success_ret.keys():
            f.write('\t%s' % tracker_name)
        f.write("\n")

        for att in metrics.keys():
            f.write("%s\t" % att)
            for ope in metrics[att]:
                if 'success' in ope:

                    tracker_names = dict()
                    success = metrics[att][ope]

                    for _tracker_name in success.keys():
                        # print(_tracker_name, success[_tracker_name])
                        tracker_names[_tracker_name] = success[_tracker_name]

                    for tracker_name in success_ret.keys():
                        f.write("%.3f\t" % tracker_names[tracker_name])

                    f.write('\n')

        f.write("\n\n\n\n\n")

        f.write("Precision\n\n")
        for tracker_name in success_ret.keys():
            f.write('\t%s' % tracker_name)
        f.write("\n")

        for att in metrics.keys():
            f.write("%s\t" % att)
            for ope in metrics[att]:
                if 'precision' in ope:

                    tracker_names = dict()
                    precision = metrics[att][ope]

                    for _tracker_name in precision.keys():
                        # print(_tracker_name, precision[_tracker_name])
                        tracker_names[_tracker_name] = precision[_tracker_name]

                    for tracker_name in success_ret.keys():
                        f.write("%.3f\t" % tracker_names[tracker_name])

                    f.write('\n')

def main():

    cfg.merge_from_file(args.config)

    dataset_dir = os.path.join(args.dataset_path, args.dataset)

    tracker_path = os.path.join(args.tracker_path, args.dataset)

    prefix = '*' if args.tracker_prefix == '+' else (args.tracker_prefix + '*')
    trackers = glob(os.path.join(args.tracker_path, args.dataset, prefix))

    trackers = [os.path.basename(x) for x in trackers]
    assert len(trackers) > 0
    args.num = min(args.num, len(trackers))

    root = os.path.realpath(os.path.join(os.path.dirname(__file__),
                            '../testing_dataset'))
    root = os.path.join(root, args.dataset)
    if 'OTB' in args.dataset:
        # dataset = OTBDataset(args.dataset, root)
        dataset = OTBDataset(args.dataset, dataset_dir, dataset_toolkit='oneshot', config=cfg)
        dataset.set_tracker(tracker_path, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)

        if args.vis:
            metrics = dict()
            for attr, videos in dataset.attr.items():
                if attr != 'ALL':
                    continue
                print(attr)
                metrics[attr] = draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)

            save_metrics(tracker_path, metrics, success_ret)

    elif 'LaSOT' == args.dataset:
        dataset = LaSOTDataset(args.dataset, dataset_dir)
        dataset.set_tracker(tracker_path, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        norm_precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_norm_precision,
                trackers), desc='eval norm precision', total=len(trackers), ncols=100):
                norm_precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret, norm_precision_ret,
                show_video_level=args.show_video_level)

        if args.vis:
            metrics = dict()
            for attr, videos in dataset.attr.items():
                if attr != 'ALL':
                    continue
                metrics[attr] = draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)

            save_metrics(tracker_path, metrics, success_ret)

    elif 'UAV123' in args.dataset or 'GOT-10k' in args.dataset:
        if 'UAV123' in args.dataset:
            dataset = UAVDataset(args.dataset, dataset_dir)
        elif 'GOT-10k' in args.dataset:
            dataset = GOT10kDataset(args.dataset, dataset_dir)
        dataset.set_tracker(tracker_path, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)

        if args.vis:
            metrics = dict()
            for attr, videos in dataset.attr.items():
                if attr != 'ALL':
                    continue
                metrics[attr] = draw_success_precision(success_ret,
                                       name=dataset.name,
                                       videos=videos,
                                       attr=attr,
                                       precision_ret=precision_ret)

            save_metrics(tracker_path, metrics, success_ret)

    elif 'NFS' in args.dataset:
        dataset = NFSDataset(args.dataset, root)
        dataset.set_tracker(dataset_dir, trackers)
        benchmark = OPEBenchmark(dataset)
        success_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_success,
                trackers), desc='eval success', total=len(trackers), ncols=100):
                success_ret.update(ret)
        precision_ret = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval_precision,
                trackers), desc='eval precision', total=len(trackers), ncols=100):
                precision_ret.update(ret)
        benchmark.show_result(success_ret, precision_ret,
                show_video_level=args.show_video_level)
    elif args.dataset in ['VOT2016', 'VOT2017', 'VOT2018', 'VOT2019']:
        dataset = VOTDataset(args.dataset, dataset_dir, config=cfg)
        dataset.set_tracker(os.path.join(args.tracker_path, args.dataset), trackers)

        ar_benchmark = AccuracyRobustnessBenchmark(dataset)
        ar_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(ar_benchmark.eval,
                trackers), desc='eval ar', total=len(trackers), ncols=100):
                ar_result.update(ret)

        tags = ['all', 'camera_motion', 'illum_change', 'motion_change', 'size_change', 'occlusion']
        benchmark = EAOBenchmark(dataset, tags=tags)
        eao_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                trackers), desc='eval eao', total=len(trackers), ncols=100):
                eao_result.update(ret)
        metrics = ar_benchmark.show_result(ar_result, eao_result,
                show_video_level=args.show_video_level)

        save_vot_metrics(tracker_path, metrics, ar_result, tags)

        # if args.vis:
        #     metrics = dict()
        #     for attr, videos in dataset.attr.items():
        
        #         metrics[attr] = draw_success_precision(success_ret,
        #                                name=dataset.name,
        #                                videos=videos,
        #                                attr=attr,
        #                                precision_ret=precision_ret)
        #
        #     save_metrics(tracker_path, metrics, success_ret)

    elif 'VOT2018-LT' == args.dataset:
        dataset = VOTLTDataset(args.dataset, root)
        dataset.set_tracker(dataset_dir, trackers)
        benchmark = F1Benchmark(dataset)
        f1_result = {}
        with Pool(processes=args.num) as pool:
            for ret in tqdm(pool.imap_unordered(benchmark.eval,
                trackers), desc='eval f1', total=len(trackers), ncols=100):
                f1_result.update(ret)
        benchmark.show_result(f1_result,
                show_video_level=args.show_video_level)


if __name__ == '__main__':
    main()
