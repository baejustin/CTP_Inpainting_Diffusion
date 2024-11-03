#-*- coding:utf-8 -*-
import os
from glob import glob
import numpy as np
import commentjson
from dataset import scale_windowing
import argparse
import torch
from torch import nn
from skimage import morphology
from scipy import ndimage
from diffusion_model.unet import create_model
from diffusion_model.trainer import GaussianDiffusion
import logging
log_mapping = {
    'debug': logging.DEBUG,
    'info': logging.INFO,
    'critical': logging.CRITICAL,
}

def collate(batch):

    batch_collated = {}
    keys = batch[0].keys()
    for key in keys:
        values = [d[key] for d in batch]

        if isinstance(values[0], torch.Tensor):
            batch_collated[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], (int, float)):
            batch_collated[key] = torch.tensor(values)
        else:
            batch_collated[key] = values
    return batch_collated

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="conifg/default_config.json") # TODO
    args = parser.parse_args()

    opt = commentjson.load(open(args.config_path, 'r'))
    logging.basicConfig(level=log_mapping[opt['loglevel']])

    gpu_ids = [int(id) for id in opt['gpu_ids'].split(',')]
    os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_ids']
    
    os.makedirs(opt['inference']['inference_output_path'], exist_ok=True)

    diffusion_model = nn.DataParallel(GaussianDiffusion(
        opt = opt,
        denoise_fn = create_model(opt),
    )).cuda()


    diffusion_model.load_state_dict(torch.load(opt['resume_weight'])['ema'])
    logging.info('Weight loaded!')

    logging.info('Setting new noise schedule for testing..')
    diffusion_model.module.set_new_noise_schedule(opt, opt['inference']['timesteps'])
    diffusion_model.eval()

    '''
    Interpolate individual low-temporal resolution sequence.
    input sequence (low_res_seq) should be a npy object of:
    - shape 8(depth) x 192(H) x 192(W) x T(length of the sequence).
    - windowed to level 40 and width 80 (0~80).
    - skull-stripped.
    '''
    inf_bs = opt['inference']['batch_size']
    inbetween_scan_gap = opt['inference']['intra_scan_gap']
    num_synthetic_scans_inbetween = inbetween_scan_gap - 1


    for seq_vol_path in glob(os.path.join(opt['data_root'],'inference', '*')):

        seq_vol_name = os.path.basename(seq_vol_path)

        logging.info('Processing sequence {}'.format(seq_vol_name))

        try:
            seq_vol_name[:-4] == '.npy'
        except:
            logging.info('Unsupported input: {}'.format(seq_vol_name))

        # load the low temporal resolution sequence (sequence of known scans.)
        low_tres_seq = np.load(seq_vol_path).astype(np.float32)
        try:
            assert np.all((low_tres_seq >= 0) & (low_tres_seq <= 80))
        except:
            logging.info('Not within the intensity range: {}'.format(seq_vol_name))


        tissue_mask = (np.sum(low_tres_seq, -1) > 0)
        tissue_mask = morphology.remove_small_objects(tissue_mask.astype(bool), 6000)
        tissue_mask = ndimage.binary_fill_holes(tissue_mask)

        print('tissue_mask', tissue_mask.shape)

         # Interleave the know scans with dummy scans.
        d, h, w, low_tres_seq_len = low_tres_seq.shape
        interpolated_seq_len = low_tres_seq_len + num_synthetic_scans_inbetween * (low_tres_seq_len - 1)
        low_tres_seq_expanded = np.zeros((d, h, w, interpolated_seq_len), dtype=low_tres_seq.dtype)
        known_scans_positions = np.arange(low_tres_seq_len) * (inbetween_scan_gap)
        low_tres_seq_expanded[..., known_scans_positions] = low_tres_seq

        low_tres_seq_interpolated = scale_windowing(opt["window_width"], opt["window_level"], low_tres_seq_expanded.copy())


        # Derive the index of the "estimated" highest peak.
        if num_synthetic_scans_inbetween == 1:
            observed_peak_idx = np.argmax(np.sum(low_tres_seq_expanded, (0,1,2)))
        else:
            top_two_peak_indices = np.argsort(np.sum(low_tres_seq_expanded, axis=tuple(range(low_tres_seq_expanded.ndim - 1))))[::-1][:2]
            observed_peak_idx = int(np.mean(top_two_peak_indices))

        
        generate_meta_dict_list = []

        for left_idx, right_idx in zip(known_scans_positions[:-1], known_scans_positions[1:]):
            gap =  (right_idx - left_idx)
            
            # Derive the contrast scenario (global context)
            if left_idx < observed_peak_idx and right_idx <= observed_peak_idx:   # Scenario 1 - Rising
                contrast_scenario = [1,0,0]
            elif left_idx >= observed_peak_idx and right_idx > observed_peak_idx: # Scenario 3 - Falling 
                contrast_scenario = [0,0,1]
            else:                                                                 # Scenario 2 - Around the peak
                contrast_scenario = [0,1,0]

            contrast_scenario = torch.Tensor(contrast_scenario).float()
            
            for local_target_idx in range(1, gap):
                # normalize the temporal distance.
                left_distance = (local_target_idx - 1) / (opt['max_tr'] - 2) 
                right_distance = (gap-local_target_idx-1) / (opt['max_tr'] - 2)
                distance_tuple = (left_distance, right_distance)

                distance_tuple = torch.Tensor(distance_tuple).float()
                
                target_idx = left_idx + local_target_idx

                left_scan = torch.tensor(low_tres_seq_interpolated[:,:,:,left_idx][np.newaxis])
                right_scan = torch.tensor(low_tres_seq_interpolated[:,:,:,right_idx][np.newaxis])

                condition_scans = torch.cat([left_scan, right_scan], dim=0)


                generate_meta_dict = {
                    'input': condition_scans, 
                    'target_idx':target_idx, 
                    'distance_tuple': distance_tuple,
                    'contrast_scenario': contrast_scenario,
                }
                generate_meta_dict_list.append(generate_meta_dict)


        for i in range(0, len(generate_meta_dict_list), inf_bs):
            inf_bs_ = min(inf_bs, len(generate_meta_dict_list)-i)

            batch = generate_meta_dict_list[i:i+inf_bs_]
            batch_collated = collate(batch)

            input_tensors_ = batch_collated['input'].cuda()
            distance_tuple_ = batch_collated['distance_tuple'].cuda()
            contrast_scenario_ = batch_collated['contrast_scenario'].cuda()

            pred_volumes = diffusion_model.module.sample(
                                        batch_size=inf_bs_, 
                                        condition_tensors = input_tensors_, 
                                        distance_tuple = distance_tuple_,
                                        contrast_scenario = contrast_scenario_ 
            )

            low_tres_seq_interpolated[..., batch_collated['target_idx']] = pred_volumes.cpu().numpy().transpose(2,3,4,0,1).squeeze()

        low_tres_seq_interpolated = ((low_tres_seq_interpolated+ 1.0) * 40).astype(np.float32)  * tissue_mask[...,np.newaxis]
        np.save(os.path.join(opt['inference']['inference_output_path'], seq_vol_name.replace('.npy','_expanded.npy')), low_tres_seq_expanded)
        np.save(os.path.join(opt['inference']['inference_output_path'], seq_vol_name.replace('.npy','interpolated_1sec.npy')), low_tres_seq_interpolated) 
