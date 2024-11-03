#-*- coding:utf-8 -*-
from torch.utils.data import Dataset
import numpy as np
import torch
import os
import pickle
import random
from tqdm import tqdm

np.random.seed(1234)


def scale_windowing(ww, wl, scan):
    '''
    Scales the given scan to (-1,1) from the windowing information.
    '''
    lower_bound = wl - (ww / 2)
    upper_bound = wl + (ww / 2)

    scan_clipped = np.clip(scan, lower_bound, upper_bound)
    scaled_scan = ((scan_clipped - lower_bound) / (upper_bound - lower_bound)) * 2 - 1

    return scaled_scan

def inverse_scale_windowing(ww, wl, scaled_scan):
    '''
    Converts a scaled scan in the range (-1, 1) back to the original intensity values.
    '''
    lower_bound = wl - (ww / 2)
    upper_bound = wl + (ww / 2)
    
    scan_clipped = lower_bound + ((scaled_scan + 1) / 2) * (upper_bound - lower_bound)
    
    return scan_clipped


class TrainDataGenerator(Dataset):
    def __init__(self, opt, phase = 'train'):

        if phase not in ['train', 'val']:
            raise NotImplementedError
        self.ww = opt["window_width"]
        self.wl = opt["window_level"]
        self.data_root = os.path.join(opt['data_root'], phase)
        self.max_tr = opt['max_tr']
        self.phase = phase

        self.preload_data_dict = {}
        for seq_vol_name in  tqdm(os.listdir(self.data_root)):
            self.preload_data_dict[seq_vol_name] = np.load(os.path.join(self.data_root, seq_vol_name))

        self.observed_peak_idx_dict = self.find_observed_peak_idx(self.preload_data_dict)


        self.data_meta_list = []
        for seq_vol_name in os.listdir(self.data_root):

            peak_idx = self.observed_peak_idx_dict[seq_vol_name]
            sequence_length = self.preload_data_dict[seq_vol_name].shape[-1]

            for i in range(sequence_length):

                for tr in range(2, self.max_tr + 1):

                    if i < sequence_length - tr:

                        left_idx = i
                        right_idx = i + tr
                        contrast_scenario = None
                        
                        if left_idx < peak_idx and right_idx <= peak_idx:   # Scenario 1 - Rising
                            contrast_scenario = [1,0,0]
                        elif left_idx >= peak_idx and right_idx > peak_idx: # Scenario 3 - Falling 
                            contrast_scenario = [0,0,1]
                        else:                                               # Scenario 2 - Around the peak
                            contrast_scenario = [0,1,0]
                        
                        for p in range(1, tr):
                            # normalize the temporal distance.
                            left_distance = (p - 1) / (self.max_tr - 2) 
                            right_distance = (tr-p-1) / (self.max_tr - 2)

                            distance_tuple = (left_distance, right_distance)

                            center_idx = i + p

                            data_dict = {
                                'seq_vol_name' : seq_vol_name,
                                'target_volume_idx' : center_idx,
                                'left_volume_idx' : left_idx,
                                'right_volume_idx' : right_idx,
                                'distance_tuple' : distance_tuple,
                                'contrast_scenario': contrast_scenario
                            }

                            self.data_meta_list.append(data_dict)



    def find_observed_peak_idx(self, preload_data_dict):
        '''
        Find the index of the frame where the contrast agent hits its peak intensity. 
        Due to limited info about the data, we simply use the overall sum of the intensity across the entire snapshot volume.
        '''
        peak_idx_dict = {}
        for seq_vol_name, np_vol in preload_data_dict.items():
            peak_idx_dict[seq_vol_name] = np.argmax(np.sum(np_vol, (0,1,2)))
        return peak_idx_dict


    def transform(self, left_volume, right_volume, center_volume):
        
        # horizontal flip
        hflip = (random.random() < 0.5) if self.phase == 'train' else False
        if hflip:
            left_volume = left_volume[:, :,  :, ::-1]
            right_volume = right_volume[:, :,  :, ::-1]
            center_volume = center_volume[:, :,  :, ::-1]

        left_volume = torch.tensor(scale_windowing(self.ww, self.wl, left_volume))
        right_volume = torch.tensor(scale_windowing(self.ww, self.wl, right_volume))
        center_volume = torch.tensor(scale_windowing(self.ww, self.wl, center_volume))
        
        return left_volume, right_volume, center_volume


    def random_sample_data(self, batch_size: int):
        indexes = np.random.randint(0, len(self), batch_size)

        condition_tensors_dict = []

        for idx in indexes:
            condition_tensors_dict.append(self.__getitem__(idx))

        return condition_tensors_dict


    def __len__(self):
        return len(self.data_meta_list)


    def __getitem__(self, index):
        
        data_meta_dict = self.data_meta_list[index]

        seq_vol_name = data_meta_dict['seq_vol_name']
        c_idx = data_meta_dict['target_volume_idx']
        l_idx = data_meta_dict['left_volume_idx']
        r_idx = data_meta_dict['right_volume_idx']

    
        distance_tuple = torch.Tensor(data_meta_dict['distance_tuple']).float()
        contrast_scenario = torch.Tensor(data_meta_dict['contrast_scenario']).float()

        full_sequence = self.preload_data_dict[seq_vol_name]
        center_volume = full_sequence[:,:,:,c_idx][np.newaxis].astype(np.float32)
        left_volume = full_sequence[:,:,:,l_idx][np.newaxis].astype(np.float32)
        right_volume = full_sequence[:,:,:,r_idx][np.newaxis].astype(np.float32)

        left_volume, right_volume, center_volume = self.transform(left_volume, right_volume, center_volume)

        condition_volumes = torch.cat([left_volume, right_volume], dim=0)


        return {
                'input': condition_volumes, 
                'target':center_volume, 
                'distance_tuple': distance_tuple,
                'contrast_scenario': contrast_scenario,
                'metadata':{
                            'name': os.path.splitext(seq_vol_name)[0],
                            'idx': 'L{},target{},R{}'.format(l_idx, c_idx, r_idx)
                            }
                }
                