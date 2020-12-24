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

from pathlib import Path

if __name__ == '__main__':
    root = Path('/data/PlateLayer/valid')
    files = root.rglob('*.jpg')
    with open('/data/PlateLayer/valid.txt', 'w') as fin:
        for f in files:
            fin.write('{}\t{}\n'.format(f, f.parent.name))
