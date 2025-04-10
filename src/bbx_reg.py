import datetime
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets_bbx import Dataset
from model import RegModel
import torchvision

def format_time(time):
    elapsed_rounded = int(round((time)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

class BoxReg():
    def __init__(self, config, world_size, rank):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.debug = False
        self.RegModel = RegModel(config, rank).to(config.DEVICE)

        self.transf = torchvision.transforms.Compose(
            [
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        # test mode
        if self.config.MODE == 2:
            self.test_dataset = Dataset(rank, config, config.TEST_LIST, config.composite_flist, config.fg_instance_flist, config.fg_shadow_flist, augment=False, training=False)
            self.val_dataset = Dataset(rank, config, config.VAL_LIST, config.composite_flist, config.fg_instance_flist, config.fg_shadow_flist, augment=False, training=False)
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
        self.RegModel.load(self.config.MODEL_LOAD)

    def save(self, max_kfiou):
        self.RegModel.save(max_kfiou)

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
        epoch = self.RegModel.iteration // len(train_loader)
        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset) // self.world_size
        max_kfiou = self.RegModel.max_kfiou

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

                self.RegModel.train()
                composite_img, fg_instance_mask, fg_instance_bbx, fg_shadow_bbx, fg_shadow_t, identifier = items
                composite_img, fg_instance_mask, fg_instance_bbx, fg_shadow_bbx, fg_shadow_t = self.cuda(composite_img, fg_instance_mask, fg_instance_bbx, fg_shadow_bbx, fg_shadow_t)

                loss_reg, logs = self.RegModel.process(composite_img, fg_instance_mask, fg_instance_bbx, fg_shadow_bbx, fg_shadow_t)
 
                # backward
                self.RegModel.backward(loss_reg)
                iteration = self.RegModel.iteration

                if iteration >= max_iteration:
                    keep_training = False
                    break

                logs = [
                    ("epoch", epoch),
                    ("iter", iteration),
                ] + logs

                # evaluate model at checkpoints
                if self.config.EVAL_INTERVAL and iteration % self.config.EVAL_INTERVAL == 0 and self.rank == 0:
                    print('\nstart eval...\n')
                    cur_kfiou = self.eval()

                    self.RegModel.iteration = iteration

                    if cur_kfiou > max_kfiou:
                        max_kfiou = cur_kfiou
                    self.save(max_kfiou)
                    print('---increase-iteration:{}'.format(iteration))
                
                # save model at checkpoints
                if self.config.SAVE_INTERVAL and iteration % self.config.SAVE_INTERVAL == 0 and self.rank == 0:
                    self.save(cur_kfiou)

                time_end_everyiter = time.time()
                time_current_iter = format_time(time_end_everyiter - time_start_everyiter)

                time_end = time.time()
                time_total = format_time(time_end - time_start)

                logs = [
                           ("iter time", time_current_iter),
                           ("total time", time_total),
                           ('best_kfiou', max_kfiou),
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

        self.RegModel.eval()

        kfiou_list = []
        loss_list = []

        with torch.no_grad():
            for items in val_loader:

                composite_img, fg_instance_mask, fg_instance_bbx, fg_shadow_bbx, fg_shadow_t, identifier = items
                composite_img, fg_instance_mask, fg_instance_bbx, fg_shadow_bbx, fg_shadow_t = self.cuda(composite_img, fg_instance_mask, fg_instance_bbx, fg_shadow_bbx, fg_shadow_t)

                loss_kfiou, KFIoU, box_points = self.RegModel.process_test(composite_img, fg_instance_mask, fg_instance_bbx, fg_shadow_bbx, fg_shadow_t)
                KFIoU = KFIoU.cpu().numpy()[0]
                loss_kfiou = loss_kfiou.cpu().numpy()
                kfiou_list.append(KFIoU)
                loss_list.append(loss_kfiou)

                if self.rank == 0:
                    print('kfiou:{}/{} loss:{}/{} num:{}/{}'.format(KFIoU, np.average(kfiou_list),
                                                            loss_kfiou, np.average(loss_list),
                                                            len(kfiou_list), len(self.val_dataset)))
                    
            if self.rank == 0:
                print('kfiou:{} loss:{}'.format(np.average(kfiou_list),
                                                np.average(loss_list)))

            return np.average(kfiou_list)

    def test(self):

        test_loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=1,
        )

        self.RegModel.eval()

        kfiou_list = []
        loss_list = []
        iteration = self.RegModel.iteration
        with torch.no_grad():
            for items in test_loader:
                gt, composite_img, fg_instance_mask, fg_instance_bbx, fg_shadow_bbx, fg_shadow_t, identifier = items
                gt, composite_img, fg_instance_mask, fg_instance_bbx, fg_shadow_bbx, fg_shadow_t = self.cuda(gt, composite_img, fg_instance_mask, fg_instance_bbx, fg_shadow_bbx, fg_shadow_t)

                loss_kfiou, KFIoU, box_points = self.RegModel.process_test(composite_img, fg_instance_mask, fg_instance_bbx, fg_shadow_bbx, fg_shadow_t)
                box_points = np.int32(box_points)

                KFIoU = KFIoU.cpu().numpy()[0]
                loss_kfiou = loss_kfiou.cpu().numpy()
                kfiou_list.append(KFIoU)
                loss_list.append(loss_kfiou)
                
                if self.rank == 0:
                    print('kfiou:{}/{} loss:{}/{} num:{}/{}'.format(KFIoU, np.average(kfiou_list),
                                                            loss_kfiou, np.average(loss_list),
                                                            len(kfiou_list), len(self.test_dataset)))
                    
            if self.rank == 0:
                print('kfiou:{} loss:{}'.format(np.average(kfiou_list),
                                                np.average(loss_list)))

            return np.average(kfiou_list)



    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)
