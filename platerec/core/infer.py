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
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image

from platerec.utils import AverageMeter
from platerec.utils import console


def infer(model, args):
    model.eval()
    transforms = T.Compose([
        T.Resize((args.imgH, args.imgW)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    end = time.time()
    batch_time = AverageMeter('Time', ':6.3f')
    files = list(Path(args.valid_data).rglob('*.jpg'))
    with torch.no_grad(), open('infer.log', 'w') as log:
        for i, fim in enumerate(files):
            image = Image.open(fim)
            image = transforms(image)
            image = image.unsqueeze(0)
            if args.gpu is not None:
                image = image.cuda()
            output = model(image)
            output = torch.argmax(output)
            batch_time.update(time.time() - end)
            end = time.time()
            console.print('Infer: {} {}'.format(batch_time, output))
            log.write('{}\t{}\n'.format(fim, output))