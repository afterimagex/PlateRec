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

import os
from pathlib import Path

import cv2
import lmdb
import numpy as np

from platerec.utils import console

PLATE_NUMBER = {
    '-': 0,
    '京': 1,
    '沪': 2,
    '津': 3,
    '渝': 4,
    '冀': 5,
    '晋': 6,
    '蒙': 7,
    '辽': 8,
    '吉': 9,
    '黑': 10,
    '苏': 11,
    '浙': 12,
    '皖': 13,
    '闽': 14,
    '赣': 15,
    '鲁': 16,
    '豫': 17,
    '鄂': 18,
    '湘': 19,
    '粤': 20,
    '桂': 21,
    '琼': 22,
    '川': 23,
    '贵': 24,
    '云': 25,
    '藏': 26,
    '陕': 27,
    '甘': 28,
    '青': 29,
    '宁': 30,
    '新': 31,
    '0': 32,
    '1': 33,
    '2': 34,
    '3': 35,
    '4': 36,
    '5': 37,
    '6': 38,
    '7': 39,
    '8': 40,
    '9': 41,
    'A': 42,
    'B': 43,
    'C': 44,
    'D': 45,
    'E': 46,
    'F': 47,
    'G': 48,
    'H': 49,
    'J': 50,
    'K': 51,
    'L': 52,
    'M': 53,
    'N': 54,
    'P': 55,
    'Q': 56,
    'R': 57,
    'S': 58,
    'T': 59,
    'U': 60,
    'V': 61,
    'W': 62,
    'X': 63,
    'Y': 64,
    'Z': 65,
    '港': 66,
    '学': 67,
    '使': 68,
    '警': 69,
    '澳': 70,
    '挂': 71,
    '军': 72,
    '北': 73,
    '南': 74,
    '广': 75,
    '沈': 76,
    '兰': 77,
    '成': 78,
    '济': 79,
    '领': 80,
    '民': 81,
    '航': 82,
    '空': 83,
}


class LMDBWriter(object):
    def __init__(self, root):
        assert not os.path.exists(root)
        self._root = root
        self._env = lmdb.open(root, map_size=1099511627776)
        self._cnt = 0

    def _write_cache(self, cache):
        with self._env.begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(k, v)

    def _check_image_is_valid(self, image_bin):
        if image_bin is None:
            return False
        image_buf = np.frombuffer(image_bin, dtype=np.uint8)
        image = cv2.imdecode(image_buf, cv2.IMREAD_GRAYSCALE)
        if image.shape[0] * image.shape[1] <= 0:
            return False
        return True

    def write(self, image_root, file_list):
        with open(file_list, 'r', encoding='utf-8') as fin:
            lines = fin.readlines()
            assert len(lines) > 0
        cache = {}
        count = 0
        for i, line in enumerate(lines):
            try:
                img_path, label = line.strip().split('\t')
                img_path = Path(image_root) / img_path

                if not img_path.exists():
                    console.print('{} does not exist.'.format(img_path))
                    continue

                with img_path.open('rb') as f:
                    image_bin = f.read()

                    if not self._check_image_is_valid(image_bin):
                        message = '{}-th {} is not a valid image.'.format(i, img_path)
                        console.print(message)
                        with (Path(self._root) / 'error_image_log.txt').open('a') as log:
                            log.write(message + '\n')
                        continue

                cache['image-{:09d}'.format(self._cnt).encode()] = image_bin
                cache['label-{:09d}'.format(self._cnt).encode()] = label.encode()

                if self._cnt % 1000 == 0:
                    self._write_cache(cache)
                    cache = {}
                    console.print('Written [{}/{}]'.format(self._cnt, len(lines)))
                self._cnt += 1
                count += 1
            except:
                console.print_exception()
        self._write_cache(cache)
        console.print('Created dataset with {} samples'.format(count))


class LMDBReader(object):
    def __init__(self, root, batch_size, shuffle=True, max_readers=32):
        self._env = lmdb.open(root, max_readers=max_readers, readonly=True, lock=False, readahead=False, meminit=False)
        assert self._env
        self._txn = self._env.begin(write=False)
        self._num_sample = self._txn.stat()['entries'] // 2
        self._indices = list(np.arange(self._num_sample))
        self._shuffle = shuffle
        self._batch_size = batch_size

    def __len__(self):
        return self._num_sample

    def __iter__(self):
        self._i = 0
        if self._shuffle:
            np.random.shuffle(self._indices)
        return self

    def __next__(self):
        return self.batches()

    def __del__(self):
        self._env.close()

    def next(self):
        image, label = None, None
        try:
            idx = self._indices[self._i]
            image = self._txn.get('image-{:09d}'.format(idx).encode())
            label = self._txn.get('label-{:09d}'.format(idx).encode())
        except Exception:
            console.print_exception()
        finally:
            self._i = (self._i + 1) % self._num_sample
        return image, label

    def batches(self):
        images, labels = [], []
        while True:
            try:
                image, label = self.next()
                assert image, label
                image = np.frombuffer(image, dtype=np.uint8)
                label = label.decode('utf-8')
                L = [0 for _ in range(8)]
                for i, l in enumerate(label):
                    if l == 'O': l = "0"
                    if l == 'I': l = "1"
                    L[i] = PLATE_NUMBER[l]
            except Exception:
                # console.print_exception()
                continue

            images.append(image)
            labels.append(np.int32(L))

            if len(images) != len(labels):
                images, labels = [], []

            if len(images) == self._batch_size:
                return images, labels


class LMDBReaderBalance(object):
    def __init__(self, root, batch_size, shuffle=True, max_reders=32, batch_ratio=None):
        files = [p.absolute().parent for p in Path(root).rglob('data.mdb')]
        files, self._sizes = self.check(files)
        if batch_ratio is not None:
            assert len(files) == len(batch_ratio)
            assert sum(batch_ratio) == 1.0
            subatch = [int(batch_size * batch_ratio[i]) for i in range(len(batch_ratio))]
            for i in range(batch_size - sum(subatch)):
                subatch[0] += 1
        else:
            subatch = [batch_size // len(files) for _ in range(len(files))]
            for i in range(batch_size % len(files)):
                subatch[i] += 1
        console.print('LEFT LMDB:', *zip(files, self._sizes, subatch), style='info')
        self._vmdb = [
            iter(LMDBReader(str(files[i]), subatch[i], shuffle, max_reders))
            for i in range(len(files))
        ]

    def __len__(self):
        return sum(self._sizes)

    def __iter__(self):
        return self

    def __next__(self):
        images, labels = [], []
        for db in self._vmdb:
            image, label = next(db)
            images += image
            labels += label
        return images, labels

    def check(self, files):
        sizes = []
        for f in files:
            db = LMDBReader(str(f), 1)
            if len(db) <= 0:
                files.remove(f)
                console.print('removed lmdb:', f, 'size is 0.', style='warning')
                continue
            sizes.append(len(db))
        return files, sizes


if __name__ == '__main__':
    # d = '京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新0123456789ABCDEFGHJKLMNPQRSTUVWXYZ港学使警澳挂军北南广沈兰成济领民航空'
    # for i, x in enumerate(d):
    #     print("\'{}\':{},".format(x, i + 1))
    # w = LMDBWriter(root='/data/PlateLayer/valid.lmdb')
    # w.write('/', '/data/PlateLayer/valid.txt')

    # w = LMDBWriter(root='/media/ps/0A9AD66165F33762/XPC/PlateCls/valid.lmdb')
    # w.write('/', '/media/ps/0A9AD66165F33762/XPC/PlateCls/valid.txt')

    lrb = LMDBReaderBalance('/media/ps/0A9AD66165F33762/XPC/plate_all/valid', 12, batch_ratio=[0.1, 0.2, 0.3, 0.2, 0.2])
    print(len(lrb))
    count = 0
    for data in lrb:
        i, l = data
        print(len(i), len(l))
        count += 1
    # print(count)
