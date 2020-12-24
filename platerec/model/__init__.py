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

from .resnet2 import resnet18


def build_model(args):
    if args.arch == 'resnet18':
        return resnet18(False, num_classes=args.num_classes)
