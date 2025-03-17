import numpy as np
import torch
from cldm.model import create_model, load_state_dict
from PIL import Image
import os
from einops import rearrange
from torch.utils.data import DataLoader
from tutorial_dataset import TestDataset
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from ber import *
import argparse
from pytorch_lightning import seed_everything

def trans_tensor2img(grid_, if_mask=False):
    if if_mask:
        grid =(grid_ > 0.5).float()
    else:
        grid = (grid_ + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
    grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
    grid = grid.numpy()
    grid = (grid * 255).astype(np.uint8)
    result_img = Image.fromarray(grid)
    return result_img



def get_args_parser():
    parser = argparse.ArgumentParser('test shadow diffusion', add_help=False)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--checkpoint_path', default='./models/pretrained_models/Shadow_cldm.ckpt', type=str)
    parser.add_argument('--gpu_id', default=1, type=int)
    parser.add_argument('--K', default=64, type=int)
    parser.add_argument('--test_dataset_path', default='./data/desobav2', type=str)
    parser.add_argument('--save_dir', default='./results', type=str)
    parser.add_argument('--model_dir', default='./models/cldm_v15.yaml', type=str)
    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    seed_everything(0)

    torch.cuda.set_device(args.gpu_id)

    model = create_model(args.model_dir).cpu()
    model.load_state_dict(load_state_dict(args.checkpoint_path, location='cuda'), strict=False)
    model = model.cuda()
    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,"gen_result"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,"gt_shadow_img"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,"gt_shadowfree_img"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,"gt_shadow_mask"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,"gt_object_mask"), exist_ok=True)

    dataset_test = TestDataset(data_file_path=args.test_dataset_path, K=args.K, device=torch.device("cuda:"+str(args.gpu_id)))
    dataloader_test = DataLoader(dataset_test, num_workers=0, batch_size=1, shuffle=False)


    for step, batch in tqdm(enumerate(dataloader_test)):
        print(step, len(dataloader_test))
        gt = rearrange(batch['gt'], 'b h w c -> b c h w')
        shadowfree_img = rearrange(batch['shadowfree_img_'], 'b h w c -> b c h w')
        shadow_mask_ = batch['shadow_mask_'].unsqueeze(0)
        object_mask_ = batch['object_mask_'].unsqueeze(0)
        img_name = batch['img_name'][0]
        pic_name, extension = os.path.splitext(img_name)    

        for key in batch.keys():
            if key == 'txt' or key == 'img_name':
                batch[key] *= 5
            elif len(batch[key].shape) == 2:
                batch[key] = batch[key].repeat(5, 1).cuda()
            elif len(batch[key].shape) == 3:
                batch[key] = batch[key].repeat(5, 1, 1).cuda()
            elif len(batch[key].shape) == 4:
                batch[key] = batch[key].repeat(5, 1, 1, 1).cuda()
    
        images = model.log_images(batch, N=5, use_x_T=True)

        for k in images:
            if isinstance(images[k], torch.Tensor):
                images[k] = images[k].detach().cpu()
                images[k] = torch.clamp(images[k], -1., 1.)
        img_to_save = []

        gt_img = trans_tensor2img(gt.squeeze(0))
        gt_object_mask_img = trans_tensor2img(object_mask_.squeeze(0),if_mask=True)
        resultlist = []
        resultlist.append(gt_img)
        resultlist.append(gt_object_mask_img)

        for i in range(5):
            result_img = F.interpolate(images['samples_cfg_scale_9.00'][i].unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True)

            result_img_pil = trans_tensor2img(result_img.squeeze(0))
            result_img_name = pic_name + '_' + str(i) + extension 
            result_img_pil.save(os.path.join(args.save_dir,'gen_result', result_img_name))

            gt_img = trans_tensor2img(gt.squeeze(0))
            gt_img_name = pic_name + '_' + str(i) + extension
            gt_img.save(os.path.join(args.save_dir, "gt_shadow_img", gt_img_name))

            shadowfree_image = trans_tensor2img(shadowfree_img.squeeze(0))
            shadowfree_img_name = pic_name + '_' + str(i) + extension
            shadowfree_image.save(os.path.join(args.save_dir, "gt_shadowfree_img", shadowfree_img_name))

            gt_shadow_mask_img = trans_tensor2img(shadow_mask_.squeeze(0),if_mask=True)
            gt_shadow_mask_img_name = pic_name + '_' + str(i) + extension 
            gt_shadow_mask_img.save(os.path.join(args.save_dir,'gt_shadow_mask', gt_shadow_mask_img_name))

            gt_object_mask_img = trans_tensor2img(object_mask_.squeeze(0),if_mask=True)
            gt_object_mask_img_name = pic_name + '_' + str(i) + extension 
            gt_object_mask_img.save(os.path.join(args.save_dir,'gt_object_mask', gt_object_mask_img_name))