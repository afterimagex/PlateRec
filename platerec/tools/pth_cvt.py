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


import torch

checkpoint = torch.load('/media/ps/0A9AD66165F33762/XPC/PlateLayer/output/best_model.pth')
torch.save(checkpoint['state_dict'], '/media/ps/0A9AD66165F33762/XPC/PlateLayer/output/model.pth')
