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

import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

from platerec.data.vmdb import LMDBReaderBalance


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, eii, num_threads, device_id, imgH, imgW):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.source = ops.ExternalSource(source=eii, num_outputs=2)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.resize = ops.Resize(resize_x=imgW, resize_y=imgH, device="gpu")
        self.brightness = ops.BrightnessContrast(device='gpu', brightness=0.5, contrast=1.5)
        self.rotate = ops.Rotate(device="gpu")
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        self.coin = ops.CoinFlip(probability=0.5)
        self.angle = ops.Uniform(range=(-10.0, 10.0))

    def define_graph(self):
        jpegs, labels = self.source()
        images = self.decode(jpegs)
        images = self.rotate(images, angle=self.angle())
        images = self.brightness(images)
        images = self.resize(images)
        images = self.cmnp(images, mirror=self.coin())
        return images, labels


class HybridValidPipe(Pipeline):
    def __init__(self, batch_size, eii, num_threads, device_id, imgH, imgW):
        super(HybridValidPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.source = ops.ExternalSource(source=eii, num_outputs=2)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.resize = ops.Resize(device="gpu", resize_x=imgW, resize_y=imgH)
        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            dtype=types.FLOAT,
            output_layout=types.NCHW,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )

    def define_graph(self):
        jpegs, labels = self.source()
        images = self.decode(jpegs)
        images = self.resize(images)
        images = self.cmnp(images)
        return images, labels


def get_imagenet_iter_dali(task, args):
    if task == 'train':
        lrw = LMDBReaderBalance(args.train_data, args.batch_size, True)
        size = len(lrw)
        pipe = HybridTrainPipe(
            batch_size=args.batch_size,
            eii=lrw,
            device_id=0,
            num_threads=args.workers,
            imgH=args.imgH, imgW=args.imgW,
        )
    else:
        lrw = LMDBReaderBalance(args.valid_data, args.batch_size, False)
        size = len(lrw)
        pipe = HybridValidPipe(
            batch_size=args.batch_size,
            eii=lrw,
            device_id=0,
            num_threads=args.workers,
            imgH=args.imgH, imgW=args.imgW,
        )
    pipe.build()
    size = size // max(1, args.world_size)
    dali_iter = DALIGenericIterator(pipe, ["images", "labels"], size,
                                    last_batch_policy=LastBatchPolicy.PARTIAL)
    return dali_iter


if __name__ == '__main__':
    g = LMDBReader('/media/ps/0A9AD66165F33762/XPC/PlateLayer/train.lmdb', batch_size=8)

    pipe = HybridTrainPipe(batch_size=8, eii=g, num_threads=1, device_id=2, imgW=128, imgH=32)
    pipe.build()
    print(pipe.epoch_size())
    # pipe_out = pipe.run()
    #
    # batch_cpu = pipe_out[0].as_cpu()
    # label_cpu = pipe_out[1]
    #
    # tmp = batch_cpu.at(0)
    # cv2.imwrite('tmp.jpg', tmp)
    # print(batch_cpu.at(2).shape)
    # print(batch_cpu.at(1).shape)
    # print(label_cpu.at(1))
