# Conditional Diffusion Model for Versatile Temporal Inpainting in 4D Cerebral CT Perfusion Imaging

This repository contains offical PyTorch implementation of **Conditional Diffusion Model for Versatile Temporal Inpainting in 4D Cerebral CT Perfusion Imaging**, MICCAI'24. 

## Abstract
Cerebral CT Perfusion (CTP) sequence imaging is a widely used modality for stroke assessment. While high temporal resolution of CT scans is crucial for accurate diagnosis, it correlates to increased radiation exposure. A promising solution is to generate synthetic CT scans to artificially enhance the temporal resolution of the sequence. We present a versatile CTP sequence inpainting model based on a conditional diffusion model, which can inpaint temporal gaps with synthetic scan to a fine 1-second interval, agnostic to both the duration of the gap and the sequence length. We achieve this by incorporating a carefully engineered conditioning scheme that exploits the intrinsic patterns of time-concentration dynamics. Our approach is much more flexible and clinically relevant compared to existing interpolation methods that either (1) lack such perfusion-specific guidances or (2) require all the known scans in the sequence, thereby imposing constraints on the length and acquisition interval. Such flexibility allows our model to be effectively applied to other tasks, such as repairing sequences with significant motion artifacts. Our model can generate accurate and realistic CT scans to inpaint gaps as wide as 8 seconds while achieving both perceptual quality and diagnostic information comparable to the ground-truth 1-second resolution sequence. Extensive experiments demonstrate the superiority of our model over prior arts in numerous metrics and clinical applicability.

## Cite this work
```
@inproceedings{10.1007/978-3-031-72069-7_7,
  title={Conditional Diffusion Model for Versatile Temporal Inpainting in 4D Cerebral CT Perfusion Imaging},
  author={Bae, Juyoung and Tong, Elizabeth and Chen, Hao},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={67--77},
  year={2024},
}
```
## Usage
Tested with Python 3.10 in a conda environment. Install the following required libraries for training and inference.
```
pip install -r requirements.txt
```
## Model weights
Our model was trained using two NIVIA H800 gpus. We release the weights of our fully trained model for out-of-the-box inference and fine-tuning. You can download our trained model weights from the link below.

[Download link for trained model weights](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jbaeaa_connect_ust_hk/EhIaze_mz-9MubdlIni1LVwB4NDkAQHn0KKFO1jB_4R0Aw?e=czivyc)

 Please set the location of the weights in the ```resume_weight``` field in the config file ```config\default_config.json```.

 ## Data for training and inference
 Please place data in ```ctp_data``` directory as follows:
```
ctp_data
└───train
    └───train_sequence_001_1s.npy
    └───train_sequence_002_1s.npy
    └───train_sequence_003_1s.npy
    └───...
└───val
    └───val_sequence_001_1s.npy
    └───val_sequence_002_1s.npy
    └───val_sequence_003_1s.npy
    └───...
└───inference
    └───inf_sequence_001_2s.npy
    └───inf_sequence_001_4s.npy
    └───inf_sequence_001_8s.npy
    └───...
```
Our model works for CT Perfusion sequences of:
* Spatial dimension: 192 × 192
* Axial number of slices: 8
* Window level and window width: 40,80 - represented as pixel value in a range of [0,80].

We provide a small sample of our private train, validation, and inference dataset. Please also find our sample inference result of 4s resolution inference sequences into a synthetic 1s sequences. 

[Download link for sample data](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jbaeaa_connect_ust_hk/EhIaze_mz-9MubdlIni1LVwB4NDkAQHn0KKFO1jB_4R0Aw?e=czivyc)

## Running our model
```
# Training
python train.py --config_path config\default_config.json

# Inference
python inference.py --config_path config\default_config.json
```
## Acknowledgements
This repository is inspired by the following repositories, and we express great gratitude for their work.
* https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement
* https://github.com/openai/guided-diffusion
* https://github.com/mobaidoctor/med-ddpm