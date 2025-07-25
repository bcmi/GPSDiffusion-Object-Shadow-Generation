import torch
from cldm.model import create_model, load_state_dict
import os
import numpy as np
import torch.nn.functional as F
from PIL import Image
import argparse
from torch.utils.data import DataLoader
from tutorial_dataset import TestDataset_single
from train_post_process import PostProcess
from tqdm import tqdm
import cv2
import random

random.seed(42)

@torch.no_grad()
def restore_img(comp_img):
    comp_img = comp_img * 255
    return comp_img

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
    parser.add_argument('--checkpoint_path_GPS', default='./models/pretrained_models/Shadow_cldm.ckpt', type=str)
    parser.add_argument('--checkpoint_path_post', default='./models/pretrained_models/Shadow_ppp.ckpt', type=str)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--save_dir_gen', default='./result_single/gen', type=str)
    parser.add_argument('--save_dir_post', default='./result_single/post', type=str)
    parser.add_argument('--model_dir', default='./models/cldm_v15.yaml', type=str)
    parser.add_argument('--shadowfree_imgs_path', default=None, type=str)
    parser.add_argument('--object_masks_path', default=None, type=str)
    return parser

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu_id)

    model = create_model(args.model_dir).cpu()
    model.load_state_dict(load_state_dict(args.checkpoint_path_GPS, location='cuda'), strict=False)
    model = model.cuda()
    model.eval()

    os.makedirs(args.save_dir_gen, exist_ok=True)

    dataset_test = TestDataset_single(shadowfree_img_path=args.shadowfree_imgs_path, object_mask_path=args.object_masks_path)
    dataloader_test = DataLoader(dataset_test, num_workers=0, batch_size=1, shuffle=False)
    for batch in dataloader_test:
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
        
        for i in range(5):
            result_img = F.interpolate(images['samples_cfg_scale_9.00'][i].unsqueeze(0), size=(256, 256), mode='bilinear', align_corners=True)
            result_img_pil = trans_tensor2img(result_img.squeeze(0))
            result_img_name = pic_name + '_' + str(i) + extension 
            result_img_pil.save(os.path.join(args.save_dir_gen, result_img_name))

    model = PostProcess(infe_steps=50).cpu()
    model.load_state_dict(load_state_dict(args.checkpoint_path_post, location='cuda'), strict=False)
    model = model.cuda()

    save_root = args.save_dir_post

    imname_total= os.listdir(args.save_dir_gen)
    filtered_imname_total = list(filter(lambda x: x.startswith(pic_name), imname_total))
    for img_name in tqdm(filtered_imname_total):
        last_underscore_index = img_name.rfind('_')
        new_img_name = img_name[:last_underscore_index] + '.jpg'
        width, height = 256, 256
        shadowfree_img_path = os.path.join(args.shadowfree_imgs_path, new_img_name)
        shadowfree_img = cv2.imread(shadowfree_img_path)
        shadowfree_img = cv2.resize(shadowfree_img, (width, height))
        object_masks_path = os.path.join(args.object_masks_path, new_img_name)
        object_mask = cv2.imread(object_masks_path, cv2.IMREAD_GRAYSCALE)
        object_mask = cv2.resize(object_mask, (width, height))
        shadowfree_img = cv2.cvtColor(shadowfree_img, cv2.COLOR_BGR2RGB)
        source = np.concatenate((shadowfree_img, object_mask[:, :, np.newaxis]), axis=-1)
        source = source.astype(np.float32) / 255.0
        shadowfree_img_ = (shadowfree_img.astype(np.float32) / 127.5) - 1.0
        device = torch.device("cuda:"+str(args.gpu_id))
        shadowfree_img_ = torch.from_numpy(shadowfree_img_.copy()).float().unsqueeze(0).to(device)
        source_ = torch.from_numpy(source.copy()).float().unsqueeze(0).to(device)
        batch = dict(hint=source_, name=img_name)
        comp_img_scaled = batch['hint'][:, :, :, :3]
        obj_mask = batch['hint'][:, :, :, 3:]
        comp_img= restore_img(comp_img_scaled)
        with torch.no_grad():
            model.eval()
            image_scaled = np.array(Image.open(os.path.join(args.save_dir_gen,img_name)).convert('RGB').resize((width, height),Image.NEAREST))
            image_scaled = torch.from_numpy(image_scaled.copy()).float().unsqueeze(0).to(device)
            image_scaled = image_scaled/127.5 - 1
            image = torch.clamp(image_scaled, -1., 1.)
            image = (image + 1.0) / 2.0
            image_256 = (image * 255).int()
        input = torch.concat([image_scaled, comp_img_scaled * 2 - 1, obj_mask], dim=-1)
        null_timeteps = torch.zeros(1, device=input.device)
        output = model.post_process_net(input.permute(0,3,1,2), timesteps=null_timeteps)
        output = output.permute(0,2,3,1)
        pred_mask = torch.greater_equal(output[:, :, :, 3], 0).int()
        adjusted_img = output[:, :, :, :3]
        adjusted_img = torch.clamp(image_scaled, -1., 1.)
        adjusted_img = (adjusted_img + 1.0) / 2.0
        adjusted_img = (adjusted_img * 255).int()
        new_composite_img = adjusted_img * pred_mask.unsqueeze(3) + (1-pred_mask.unsqueeze(3)) * comp_img
        filename = img_name
        os.makedirs(save_root, exist_ok=True)
        save_path = os.path.join(save_root, filename)
        save_tuned = Image.fromarray(np.array(new_composite_img.squeeze(0).detach().cpu(), dtype=np.uint8))

        # save_tuned_resized = save_tuned.resize((512, 512))
        save_tuned.save(os.path.join(save_root, img_name))
