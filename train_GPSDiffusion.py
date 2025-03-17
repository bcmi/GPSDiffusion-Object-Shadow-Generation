import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import TrainDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import torch
import argparse
import os
from pytorch_lightning.callbacks import ModelCheckpoint

def get_args_parser():
    parser = argparse.ArgumentParser('train shadow diffusion', add_help=False)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--logger_freq', default=300, type=int)
    parser.add_argument('--sd_locked', action='store_true', default=True)
    parser.add_argument('--only_mid_control', action='store_true', default=False)
    parser.add_argument('--resume_path', default='./models/pretrained_models/Shadow_cldm.ckpt', type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--K', default=64, type=int)
    parser.add_argument('--train_dataset_path', default='./data/desobav2', type=str)
    parser.add_argument('--model_dir', default='./models/cldm_v15.yaml', type=str)
    return parser

parser = get_args_parser()
args = parser.parse_args()

model = create_model(args.model_dir).cpu()
if os.path.exists(args.resume_path):
    model_weight = load_state_dict(args.resume_path, location='cpu')
    model.load_state_dict(model_weight, strict=False)
model.learning_rate = args.learning_rate
model.sd_locked = args.sd_locked
model.only_mid_control = args.only_mid_control

print(args.K)
dataset = TrainDataset(data_file_path=args.train_dataset_path, K=args.K)
dataloader = DataLoader(dataset, num_workers=8, batch_size=args.batch_size, shuffle=True)

checkpoint_callback = ModelCheckpoint(
    dirpath='models/ckpts', 
    filename='epoch_{epoch:02d}',  
    save_top_k=-1, 
    save_weights_only=True, 
    every_n_epochs=5,
)
logger = ImageLogger(batch_frequency=args.logger_freq)
trainer = pl.Trainer(accelerator='ddp', gpus=[2, 3], precision=32, callbacks=[checkpoint_callback, logger], max_epochs=30)
trainer.fit(model, dataloader)
