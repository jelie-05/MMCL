#!usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from PIL import Image
from .bin2depth import get_velo_points


class Kittiloader(object):
    """
    param kittiDir: KITTI dataset root path, e.g. ~/data/kitti/
    param mode: 'train', 'test' or 'val'
    param cam: camera id. 2 represents the left cam, 3 represents the right one
    """
    def __init__(self, kittiDir, mode, cam=2):
        self.mode = mode
        self.cam = cam
        self.files = []
        self.kitti_root = kittiDir

        # read filenames files
        currpath = os.path.dirname(os.path.realpath(__file__))
        filepath = currpath + '/filenames/eigen_{}_files.txt'.format(self.mode)
        with open(filepath, 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:
                if len(data) == 0:
                    continue
                data_info = data.split(' ')

                self.files.append({
                    "l_rgb": data_info[0],
                    "r_rgb": data_info[1],
                    "cam_intrin": data_info[2],
                    "depth": data_info[3]
                })

    def data_length(self):
        return len(self.files)

    def _check_path(self, filename, err_info):
        file_path = os.path.join(self.kitti_root, filename)
        assert os.path.exists(file_path), err_info
        return file_path

    def _read_data(self, item_files):
        l_rgb_path = self._check_path(item_files['l_rgb'], err_info="Panic::Cannot find Left Image. Filename: {}".format(item_files['l_rgb']))
        cam_path = self._check_path(item_files['cam_intrin'], err_info="Panic::Cannot find Camera Infos. Filename: {}".format(item_files['l_rgb']))
        depth_path = self._check_path(item_files['depth'], err_info="Panic::Cannot find depth file. Filename: {}".format(item_files['l_rgb']))

        l_rgb = Image.open(l_rgb_path).convert('RGB')
        w, h = l_rgb.size
        cam2cam, velo2cam, velo = get_velo_points(cam_path, depth_path, [h, w], cam=self.cam, interp=True, vel_depth=True)

        data = {}
        data['left_img'] = l_rgb
        data['cam2cam'] = cam2cam.astype(np.float32)
        data['velo2cam'] = velo2cam.astype(np.float32)
        # data['loc_l_rgb'] = item_files['l_rgb']
        data['velo'] = velo.astype(np.float32)
        return data

    def load_item(self, idx, interp_method='linear'):
        """
        load an item for training or test
        interp_method can be selected from [linear', 'nyu']
        """
        item_files = self.files[idx]
        data_item = self._read_data(item_files)

        return data_item
