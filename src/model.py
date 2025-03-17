import os
import torch
import torch.nn as nn
import torch.optim as optim
from src.bbx_network import RegNetwork, ClsNetwork
from src.loss import KFLoss
import numpy as np
from torch.optim import lr_scheduler
import cv2

class BaseModel(nn.Module):
    def __init__(self, name, config, rank):
        super(BaseModel, self).__init__()
        self.name = name
        self.rank = rank
        self.config = config
        self.iteration = 0
        self.max_acc = 0
        self.max_kfiou = 0
        self.model_save = config.PATH
        
    def load(self, type):
        self.weights_path =self.model_save + '/' + type + '.pth'
        if os.path.exists(self.weights_path) and self.rank == 0:
            print('Loading %s generator...' % self.name)

            if torch.cuda.is_available():
                data = torch.load(self.weights_path)
            else:
                data = torch.load(self.weights_path, map_location=lambda storage, loc: storage)

            self.net.load_state_dict(data['net'])
            self.iteration = data['iteration']
            self.max_acc = data['max_acc']

    def save(self, max_acc):

        if len(self.config.GPU) > 1:
            net_param = self.net.module.state_dict()
            print('save...multiple GPU')
        else:
            net_param = self.net.state_dict()
            print('save...single GPU')

        torch.save({
            'iteration': self.iteration,
            'net': net_param,
            'max_acc': max_acc
        }, os.path.join(self.model_save, '{}_{}.pth'.format(self.iteration, self.name)))


        print('\nsaving %s...\n' % self.name)

class ClsModel(BaseModel):
    def __init__(self, config, rank):
        super(ClsModel, self).__init__('ClsModel', config, rank)

        net = ClsNetwork(num_classes=256)
        loss_cross = nn.CrossEntropyLoss()

        self.add_module('net', net)
        self.add_module('loss_cross', loss_cross)

        self.optimizer = optim.Adam(
            params=net.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2))

    def process(self, composite_images, fg_instance_mask, label):
        self.iteration = self.iteration + 1

        # zero optimizers
        self.optimizer.zero_grad()

        # process outputs
        pred_label = self(composite_images, fg_instance_mask)
        pred_label = pred_label.float().requires_grad_() 

        # CLS loss
        loss_cls = self.loss_cross(pred_label, label)

        lr = self.optimizer.param_groups[0]['lr']
        # create logs
        logs = [
            ("loss_cls", loss_cls.item()),
            ("lr", lr),
        ]

        return loss_cls, logs
    
    def process_test(self, n, correct, composite_images, fg_instance_mask, label):
        self.iteration = self.iteration + 1
        pred_label = self(composite_images, fg_instance_mask)
        _, topk_indices = torch.topk(pred_label, 64, dim=1)
        for i in range(len(topk_indices)):
            correct += (topk_indices[i] == label).sum()
        n += 1
        
        return correct, n
    
    def forward(self, composite_images, fg_instance_mask):
        inputs = torch.cat((composite_images, fg_instance_mask), dim=1)
        pred_label = self.net(inputs)
        return pred_label

    def backward(self, loss_cls):

        loss_cls.backward()
        self.optimizer.step()

class RegModel(BaseModel):
    def __init__(self, config, rank):
        super(RegModel, self).__init__('RegModel', config, rank)

        net = RegNetwork()
        loss_kfiou = KFLoss(fun='exp')

        self.add_module('net', net)
        self.add_module('loss_kfiou', loss_kfiou)

        self.optimizer = optim.Adam(
            params=net.parameters(),
            lr=float(config.LR),
            betas=(config.BETA1, config.BETA2)
        )

    def process(self, composite_images, fg_instance_mask, fg_instance_bbx, fg_shadow_bbx, fg_shadow_t):
        self.iteration = self.iteration + 1

        # zero optimizers
        self.optimizer.zero_grad()

        # process outputs
        pred_t = self(composite_images, fg_instance_mask)

        # KFIOU loss
        pred_bbx = pred_t.new(pred_t.shape)
        pred_bbx[:,0] = pred_t[:,0] * fg_instance_bbx[:,2] + fg_instance_bbx[:,0]
        pred_bbx[:,1] = pred_t[:,1] * fg_instance_bbx[:,3] + fg_instance_bbx[:,1]
        pred_bbx[:,2] = fg_instance_bbx[:,2] * torch.exp(pred_t[:,2])
        pred_bbx[:,3] = fg_instance_bbx[:,3] * torch.exp(pred_t[:,3])
        pred_bbx[:,4] = pred_t[:,4] * 180 / np.pi +fg_instance_bbx[:,4]
        loss_reg, _ = self.loss_kfiou(pred_t, fg_shadow_t, pred_bbx, fg_shadow_bbx)

        lr = self.optimizer.param_groups[0]['lr']
        # create logs
        logs = [
            ("loss_reg", loss_reg.item()),
            ("lr", lr),
        ]

        return loss_reg, logs
    
    def process_test(self, composite_images, fg_instance_mask, fg_instance_bbx, fg_shadow_bbx, fg_shadow_t):
        self.iteration = self.iteration + 1
        # process outputs
        pred_t = self(composite_images, fg_instance_mask)

        # KFIOU loss
        pred_bbx = pred_t.new(pred_t.shape)
        pred_bbx[:,0] = pred_t[:,0] * fg_instance_bbx[:,2] + fg_instance_bbx[:,0]
        pred_bbx[:,1] = pred_t[:,1] * fg_instance_bbx[:,3] + fg_instance_bbx[:,1]
        pred_bbx[:,2] = fg_instance_bbx[:,2] * torch.exp(pred_t[:,2])
        pred_bbx[:,3] = fg_instance_bbx[:,3] * torch.exp(pred_t[:,3])
        pred_bbx[:,4] = pred_t[:,4] * 180 / np.pi +fg_instance_bbx[:,4]

        if pred_bbx[:,4] > 0:
                temp = pred_bbx[:,2]
                pred_bbx[:,2] = pred_bbx[:,3]
                pred_bbx[:,3] = temp
                pred_bbx[:,4] = pred_bbx[:,4] - 90
        x, y, w, h, theta = pred_bbx[:,0], pred_bbx[:,1], pred_bbx[:,2], pred_bbx[:,3], pred_bbx[:,4]
        box = ((x, y), (w, h), theta)
        box_points = cv2.boxPoints(box)

        loss_kfiou, KFIoU = self.loss_kfiou(pred_t, fg_shadow_t, pred_bbx, fg_shadow_bbx)

        return loss_kfiou, KFIoU, box_points
    
    def forward(self, composite_images, fg_instance_mask):
        inputs = torch.cat((composite_images, fg_instance_mask), dim=1)
        pred_t = self.net(inputs)
        return pred_t

    def backward(self, loss_reg):

        loss_reg.backward()
        self.optimizer.step()