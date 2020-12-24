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

import itertools


def value():
    for i in range(100):
        yield i


def lmdb_reader_wapper():
    while True:
        chain = itertools.chain(*[value(), value()])
        for x in chain:
            print(x)
            yield x


dp = lmdb_reader_wapper()
for x_ in dp:
    pass
    # print(x)

for x_ in dp:
    print(x_)
