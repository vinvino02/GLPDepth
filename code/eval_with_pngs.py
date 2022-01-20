# Copyright (C) 2019 Jin Han Lee
#
# This file is a part of BTS.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

from __future__ import absolute_import, division, print_function

import os
import argparse
import cv2
import numpy as np
from tqdm import trange
from PIL import Image

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='Evaluate Depth Estimation Results', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--pred_path',           type=str,   help='path to the prediction results in png', required=True)
parser.add_argument('--gt_path',             type=str,   help='root path to the groundtruth data', required=True)
parser.add_argument('--dataset',             type=str,   help='dataset to test on, nyudepthv2 or kitti', default='nyudepthv2',
                    choices=['kitti', 'nyudepthv2'])
parser.add_argument('--eigen_crop',                      help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop',                       help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--min_depth_eval',      type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',      type=float, help='maximum depth for evaluation', default=10.0)
parser.add_argument('--do_kb_crop',                      help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--split',               type=str,   help='split test on kitti', default='eigen_benchmark')
args = parser.parse_args()


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25 ).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred)**2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


def test():
    global gt_depths, missing_ids, pred_filenames
    gt_depths = []
    missing_ids = set()
    pred_filenames = [i for i in os.listdir(args.pred_path) if i.split('.')[-1] in ['jpg', 'png']]    
    num_test_samples = len(pred_filenames)    
    pred_depths = []

    print("Read pred images")

    for i in trange(num_test_samples):
        pred_depth_path = os.path.join(args.pred_path, pred_filenames[i])
        pred_depth = np.array(Image.open(pred_depth_path))
        
        if pred_depth is None:
            print('Missing: %s ' % pred_depth_path)
            missing_ids.add(i)
            continue

        if args.dataset == 'nyudepthv2':
            pred_depth = pred_depth.astype(np.float32) / 1000.0
        else:
            pred_depth = pred_depth.astype(np.float32) / 256.0

        pred_depths.append(pred_depth)        

    print('Raw png files reading done')
    print('Evaluating {} files'.format(len(pred_depths)))

    if args.dataset == 'kitti':
        for t_id in range(num_test_samples):
            file_dir = pred_filenames[t_id].split('.')[0]
            filename = file_dir.split('_')[-1]
            directory = file_dir.replace('_' + filename, '')        

            if args.split == 'eigen_benchmark':
                gt_depth_path = os.path.join(args.gt_path, 'data_depth_annotated', 'train', directory, 
                                             'proj_depth/groundtruth/image_02', filename + '.png')
            
                if not os.path.isfile(gt_depth_path):
                    gt_depth_path = gt_depth_path.replace("train", "val")
            else:                
                gt_depth_path = os.path.join(args.gt_path, '_'.join(file_dir.split('_')[0:3]), directory, 
                                             'velodyne_points/data', filename + '.png')            

            depth = cv2.imread(gt_depth_path, -1)
            if depth is None:
                print('Missing: %s ' % gt_depth_path)
                missing_ids.add(t_id)
                continue

            depth = depth.astype(np.float32) / 256.0
            gt_depths.append(depth)

    elif args.dataset == 'nyudepthv2':
        for t_id in range(num_test_samples):
            file_dir = pred_filenames[t_id].split('.')[0]            
            filename = file_dir.split('_')[-1]
            directory = file_dir.replace('_rgb_'+file_dir.split('_')[-1], '')
            
            gt_depth_path = os.path.join(args.gt_path, 'official_splits/test', directory,
                                         'sync_depth_' + filename + '.png')
            
            depth = cv2.imread(gt_depth_path, -1)
            if depth is None:
                print('Missing: %s ' % gt_depth_path)
                missing_ids.add(t_id)
                continue
            
            depth = depth.astype(np.float32) / 1000.0
            gt_depths.append(depth)
        
    print('GT files reading done')
    print('{} GT files missing'.format(len(missing_ids)))

    print('Computing errors')
    eval(pred_depths)

    print('Done.')

def eval(pred_depths):
    num_samples = len(pred_depths)
    pred_depths_valid = []

    i = 0
    for t_id in range(num_samples):
        if t_id in missing_ids:
            continue

        pred_depths_valid.append(pred_depths[t_id])

    num_samples = num_samples - len(missing_ids)

    silog = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)
    rms = np.zeros(num_samples, np.float32)
    log_rms = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    d1 = np.zeros(num_samples, np.float32)
    d2 = np.zeros(num_samples, np.float32)
    d3 = np.zeros(num_samples, np.float32)

    for i in trange(num_samples):

        gt_depth = gt_depths[i]
        pred_depth = pred_depths_valid[i]

        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        if args.do_kb_crop:
            height, width = gt_depth.shape
            top_margin = int(height - 352)
            left_margin = int((width - 1216) / 2)
            gt_cropped = gt_depth[top_margin:top_margin + 352, left_margin:left_margin + 1216]
            gt_depth = gt_cropped
        
        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)
        eval_mask = None

        if args.garg_crop or args.eigen_crop:
            gt_height, gt_width = gt_depth.shape
            eval_mask = np.zeros(valid_mask.shape)

            if args.garg_crop:
                eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
                int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

            elif args.eigen_crop:
                eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
                int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1

        if args.dataset == 'nyudepthv2':
            eval_mask = np.zeros(valid_mask.shape)
            eval_mask[45:471, 41:601] = 1

        if eval_mask is not None:
            valid_mask = np.logical_and(valid_mask, eval_mask)
        
        silog[i], log10[i], abs_rel[i], sq_rel[i], rms[i], log_rms[i], d1[i], d2[i], d3[i] = compute_errors(
            gt_depth[valid_mask], pred_depth[valid_mask])

    print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format(
        'd1', 'd2', 'd3', 'AbsRel', 'SqRel', 'RMSE', 'RMSElog', 'SILog', 'log10'))
    print("{:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}, {:7.3f}".format(
        d1.mean(), d2.mean(), d3.mean(),
        abs_rel.mean(), sq_rel.mean(), rms.mean(), log_rms.mean(), silog.mean(), log10.mean()))

    return silog, log10, abs_rel, sq_rel, rms, log_rms, d1, d2, d3


def main():    
    if args.dataset == 'kitti':
        args.do_kb_crop = True

    test()


if __name__ == '__main__':
    main()



