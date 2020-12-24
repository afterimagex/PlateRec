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

from platerec.core.valid import accuracy
from platerec.utils import AverageMeter, ProgressMeter


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc_ed = AverageMeter('Acc@ED', ':6.2f')
    acc_fu = AverageMeter('Acc@FU', ':6.2f')
    progress = ProgressMeter(train_loader.size // args.batch_size, batch_time, data_time, losses, acc_ed, acc_fu,
                             prefix="Epoch: [{}]".format(epoch))

    # adjust_learning_rate(optimizer, epoch, args)
    # switch to train mode
    model.train()

    end = time.time()
    for i, data in enumerate(train_loader):
        # measure data loading time
        images = data[0]['images'].cuda()
        target = data[0]['labels'].squeeze(-1).cuda().long()
        data_time.update(time.time() - end)

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

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    train_loader.reset()
