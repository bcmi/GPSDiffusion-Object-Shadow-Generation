import datetime
import time
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.datasets_cls import Dataset
from model import ClsModel
import torchvision
import sys
import cv2
import torch.nn.functional as F

def format_time(time):
    elapsed_rounded = int(round((time)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


class MaskCls():
    def __init__(self, config, world_size, rank):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.debug = False
        self.ClsModel = ClsModel(config, rank).to(config.DEVICE)

        self.transf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(rank, config, config.TEST_LIST, config.composite_flist, config.fg_instance_flist, config.fg_shadow_flist, augment=False, training=False)
            if rank == 0:
                print('test dataset:'.format(len(self.test_dataset)))
        else:
            self.train_dataset = Dataset(rank, config, config.TRAIN_LIST, config.composite_flist, config.fg_instance_flist, config.fg_shadow_flist, augment=True, training=True)
            self.val_dataset = Dataset(rank, config, config.VAL_LIST, config.composite_flist, config.fg_instance_flist, config.fg_shadow_flist, augment=False, training=False)
            self.sample_iterator = self.val_dataset.create_iterator(config.SAMPLE_SIZE)
            if self.rank == 0:
                print('train dataset:{}'.format(len(self.train_dataset)))
                print('eval dataset:{}'.format(len(self.val_dataset)))
        if len(self.config.GPU) > 1:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank)

        if config.DEBUG is not None and config.DEBUG != 0:
            self.debug = True


    def load(self):
        self.ClsModel.load(self.config.MODEL_LOAD)

    def save(self, max_acc):
        self.ClsModel.save(max_acc)

    def train(self):
        if len(self.config.GPU) > 1:
            batchsize = self.config.BATCH_SIZE // self.world_size
            train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=batchsize,
                num_workers=12,
                drop_last=False,
                shuffle=False,
                pin_memory=True,
                sampler=self.train_sampler
            )
        else:
            train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.config.BATCH_SIZE,
                num_workers=12,
                drop_last=False,
                shuffle=True,
                pin_memory=True,
            )
        epoch = self.ClsModel.iteration // len(train_loader)
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset) // self.world_size
        max_acc = self.ClsModel.max_acc

        if total == 0 and self.rank == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        time_start = time.time()

        while(keep_training):
            epoch = epoch + 1
            
            if len(self.config.GPU) > 1:
                self.train_sampler.set_epoch(epoch  + 1)
            time_start_everyiter = time.time()
            if self.rank == 0:
                print('\n\nTraining epoch: %d' % epoch)
            for items in train_loader:

                self.ClsModel.train()
                composite_img, fg_instance_mask, label = self.cuda(*items)

                loss_cls, logs = self.ClsModel.process(composite_img, fg_instance_mask, label)
 
                # backward
                self.ClsModel.backward(loss_cls)
                iteration = self.ClsModel.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                # evaluate model at checkpoints
                if iteration > 1 and self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0 and self.rank == 0:
                    print('\nstart eval...\n')
                    cur_acc = self.eval()
                    self.ClsModel.iteration = iteration

                    if cur_acc > max_acc:
                        max_acc = cur_acc
                        self.save(max_acc)
                    print('---increase-iteration:{}'.format(iteration))

                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0 and self.rank == 0:
                    self.save(max_acc)

                time_end_everyiter = time.time()
                time_current_iter = format_time(time_end_everyiter - time_start_everyiter)

                time_end = time.time()
                time_total = format_time(time_end - time_start)

                logs = [
                           ("iter time", time_current_iter),
                           ("total time", time_total),
                           ('best_acc', max_acc),
                       ] + logs
                if self.rank == 0:
                    print(logs)
        if self.rank == 0:
            print('\nEnd training....')

    def eval(self):
        val_loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=1,
            drop_last=False,
            shuffle=False
        )

        self.ClsModel.eval()

        correct = 0
        n = 0
        
        iteration = self.ClsModel.iteration
        with torch.no_grad():
            for items in val_loader:

                composite_img, fg_instance_mask, label = self.cuda(*items)

                correct, n = self.ClsModel.process_test(n, correct, composite_img, fg_instance_mask, label)

                if self.rank == 0:
                    print('acc:{} num:{} {}/{}'.format( correct / n,
                                                        correct, 
                                                        n, len(self.val_dataset)))
                    
            if self.rank == 0:
                print('iteration:{} ave_acc:{}'.format( iteration, 
                                                        correct / len(self.val_dataset),))

            return correct / len(self.val_dataset)

    def test(self):

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        self.ClsModel.eval()

        correct = 0
        n = 0
        iteration = self.ClsModel.iteration
        with torch.no_grad():
            for items in test_loader:
                composite_img, fg_instance_mask, label = self.cuda(*items)

                correct, n = self.ClsModel.process_test(n, correct, composite_img, fg_instance_mask, label)

                if self.rank == 0:
                    print('acc:{} num:{} {}/{}'.format( correct / n,
                                                        correct, 
                                                        n, len(self.test_dataset)))
                    
            if self.rank == 0:
                print('iteration:{} ave_acc:{}'.format( iteration, 
                                                        correct / len(self.test_dataset),
                                                        ))

            return correct / len(self.test_dataset)


    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
