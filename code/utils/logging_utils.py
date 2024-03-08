import os
import cv2
import sys
import time
import numpy as np

import torch

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 30.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, epochs, cur_epoch, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time
    remain_time = step_time * (total - current) + \
        (epochs - cur_epoch) * step_time * total

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    L.append(' | Rem: %s' % format_time(remain_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes).zfill(2) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf).zfill(2) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis).zfill(3) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


def display_result(result_dict):
    line = "\n"
    line += "=" * 100 + '\n'
    for metric, value in result_dict.items():
        line += "{:>10} ".format(metric)
    line += "\n"
    for metric, value in result_dict.items():
        line += "{:10.4f} ".format(value)
    line += "\n"
    line += "=" * 100 + '\n'

    return line
    

def save_images(pred, save_path):
    if len(pred.shape) > 3:
        pred = pred.squeeze()

    if isinstance(pred, torch.Tensor):
        pred = pred.cpu().numpy().astype(np.uint8)

    if pred.shape[0] < 4:
        pred = np.transpose(pred, (1, 2, 0))
    cv2.imwrite(save_path, pred, [cv2.IMWRITE_PNG_COMPRESSION, 0])


def check_and_make_dirs(paths):
    if not isinstance(paths, list):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(path)

def log_args_to_txt(log_txt, args):
    if not os.path.exists(log_txt):
        with open(log_txt, 'w') as txtfile:
            args_ = vars(args)
            args_str = ''
            for k, v in args_.items():
                args_str = args_str + str(k) + ':' + str(v) + ',\t\n'
            txtfile.write(args_str + '\n')