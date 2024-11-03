#-*- coding:utf-8 -*-
import os 
import json
import commentjson
import argparse
from diffusion_model.unet import create_model
from diffusion_model.runner import GaussianDiffusion, Runner
from dataset import TrainDataGenerator 
from initialize import init_network


parser = argparse.ArgumentParser()
parser.add_argument('--config_path', type=str, default="conifg/default_config.json") # TODO
args = parser.parse_args()

opt = commentjson.load(open(args.config_path, 'r'))

os.makedirs(opt['experiment_output_path'], exist_ok=True)

with open(os.path.join(opt['experiment_output_path'],'config_cached.json'),'w') as f:
    json.dump(opt, f, indent=2)

os.environ['CUDA_VISIBLE_DEVICES'] = opt['gpu_ids']

diffusion_model = GaussianDiffusion(
    opt = opt,
    denoise_fn = create_model(opt),
).cuda()

runner = Runner(
    opt = opt,
    diffusion_model= init_network(diffusion_model, opt['training']['weight_init']),
    train_dataset=TrainDataGenerator(opt, phase='train'),
    val_dataset = TrainDataGenerator(opt, phase='val')
)

if len(opt['resume_weight']) > 0:
    saved_weight_path = opt['resume_weight']
    runner.load(saved_weight_path)


if __name__ == '__main__':
    runner.train()