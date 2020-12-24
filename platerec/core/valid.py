# -*- coding: utf-8 -*-
# ------------------------------------------------------------
# Copyright (C) 2020-Present, Pvening, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

import time

import torch
import torch.nn.functional as F

from platerec.utils import AverageMeter, ProgressMeter


def accuracy(output, target):
    '''
    output (n, num_class, 8)
    '''
    with torch.no_grad():
        probs, preds = F.softmax(output, dim=1).max(dim=1)
        # print(preds.shape, target.shape)
        correct = preds.eq(target).float()
        ed = correct.view(-1).mean(0)
        fu = correct.prod(1).mean(0)
        return ed, fu


def valid(valid_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_ed = AverageMeter('Acc@ED', ':6.2f')
    acc_fu = AverageMeter('Acc@FU', ':6.2f')
    progress = ProgressMeter(valid_loader.size // args.batch_size, batch_time, losses, acc_ed, acc_fu, prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(valid_loader):
            images = data[0]['images'].cuda()
            target = data[0]['labels'].squeeze(-1).cuda().long()
            # if args.gpu is not None:
            #     images = images.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            ed, fu = accuracy(output, target)
            losses.update(loss.item(), images.size(0))
            acc_ed.update(ed.item(), images.size(0))
            acc_fu.update(fu.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@ED {acc_ed.avg:.3f} Acc@FU {acc_fu.avg:.3f}'.format(acc_ed=acc_ed, acc_fu=acc_fu))
    valid_loader.reset()

    return acc_fu.avg
