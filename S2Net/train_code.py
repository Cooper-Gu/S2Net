import os
import torch
import glob
import torch.utils.data as data
from tqdm import tqdm
import numpy as np
from torchvision.transforms import transforms
from kits import metrics
from kits import SegMetrics
from kits import configure_loss
from kits import LR_Scheduler
from kits import Saver
from kits import TensorboardSummary
from data import get_segmentation_dataset

sample=[]

class Trainer(object):
    def __init__(self, args, model):
        self.args = args
        self.device = torch.device(args.device)
        self.model = model
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()

        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        train_img_paths = glob.glob("/home/zxk/data/ISLES_npy/train_data/image/image/*")
        train_gt = glob.glob("/home/med_bci/data/datasets/ISLES_2022/ISLES_npy/train_data/mask/*")

        val_img_paths = glob.glob("/home/med_bci/data/datasets/ISLES_2022/ISLES_npy/val_data/image/*")
        val_gt = glob.glob("/home/med_bci/data/datasets/ISLES_2022/ISLES_npy/val_data/mask/*")

        test_img_paths = glob.glob("/home/med_bci/data/datasets/ISLES_2022/ISLES_npy/test_data/image/*")
        test_gt = glob.glob("/home/med_bci/data/datasets/ISLES_2022/ISLES_npy/test_data/mask/*")

        x_transforms = transforms.ToTensor()
        y_transforms = transforms.ToTensor()

        train_dataset = get_segmentation_dataset(args.dataset, img_paths=train_img_paths, mask_paths=train_gt, x_transform=x_transforms, y_transform=y_transforms)
        val_dataset = get_segmentation_dataset(args.dataset, img_paths=val_img_paths, mask_paths=val_gt,x_transform=x_transforms, y_transform=y_transforms)
        test_dataset = get_segmentation_dataset(args.dataset, img_paths=test_img_paths, mask_paths=test_gt,x_transform=x_transforms, y_transform=y_transforms)

        self.train_loader = data.DataLoader(dataset=train_dataset,
                                            batch_size=args.batch_size,
                                            num_workers=4,
                                            shuffle=False,
                                            pin_memory=True,
                                            drop_last=True)
        self.val_loader = data.DataLoader(dataset=val_dataset,
                                          batch_size=args.batch_size,
                                          num_workers=4,
                                          pin_memory=True)
        self.test_loader = data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          num_workers=4,
                                          pin_memory=True)

        self.optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                                         weight_decay=args.weight_decay)

        # Define criterion
        self.criterion1 = configure_loss('dice')
        self.criterion2 = configure_loss('bce')

        # # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Define Evaluator
        self.evaluator1 = SegMetrics(2)
        self.evaluator2 = SegMetrics(2)
        self.evaluator3 = SegMetrics(2)

        self.evaluator1_1 = SegMetrics(2)
        self.evaluator1_2 = SegMetrics(2)
        self.evaluator1_3 = SegMetrics(2)

        self.evaluator2_1 = SegMetrics(2)
        self.evaluator2_2 = SegMetrics(2)
        self.evaluator2_3 = SegMetrics(2)

        # Resuming checkpoint
        self.train_loss = float('inf')
        self.best_pred = 0.0


        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        num_img_tr = len(self.train_loader)
        iter=0
        #image=[]
        #sample=[]
        #image = sample.to(self.device)
        #target = sample.to(self.device)


        for iter, (image, target, file_name) in enumerate(self.train_loader):
            image = image.to(self.device)
            target = target.to(self.device)
            self.scheduler(self.optimizer, iter, epoch, self.best_pred)
            self.optimizer.zero_grad()
            if self.args.model == 'S2Net':
                output = self.model(image)
            # elif self.args.model == 'UNet':
            #     output = self.model(image)
            # elif self.args.model == 'FFNet':
            #     output, output_m1, output_m2 = self.model(image)
            # elif self.args.model == 'ACMINet':
            #     output, deeps = self.model(image)
            else:
                raise RuntimeError("modalities error, chossen{}".format(self.args.modalities))

            output = torch.sigmoid(output)
            dice_loss = self.criterion1(output, target)
            bce_loss = self.criterion2(output, target)
            loss = dice_loss + bce_loss

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            if iter % 10 == 9:
                print('Iter {} of {} Train loss {:.4f}'.format(iter + 1 , len(self.train_loader), (train_loss / (iter + 1))))
            self.writer.add_scalar('train/total_loss_iter', loss.item(), iter + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if iter % (num_img_tr // 10) == 0:
                global_step = iter + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        #print('[Epoch: %d, numImages: %5d]' % (epoch, iter * self.args.batch_size + np.array(image).shape[0]))
        print('[Epoch: %d, numImages: %5d]' % (epoch, iter * self.args.batch_size))
        print('Loss: %.3f' % (train_loss / (iter + 1)))

        if train_loss < self.train_loss:
            self.train_loss = train_loss
            is_best = True
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
                'train_loss': self.train_loss
            }, is_best)

    def validation(self, epoch):
        self.model.eval()
        test_loss = 0.0
        iter=0
        #image = sample.to(self.device)
        #target = sample.to(self.device)       

        hausdorff_sum = 0.0
        iou_sum = 0.0


        for iter, sample in enumerate(self.val_loader):
            image = sample[0].to(self.device)
            target = sample[1].to(self.device)
            fn = sample[2]

            with torch.no_grad():
                if self.args.model == 'S2Net':
                    output = self.model(image)
                # elif self.args.model == 'UNet':
                #     output = self.model(image)
                # elif self.args.model == 'FFNet':
                #     output, output_m1, output_m2 = self.model(image)
                # elif self.args.model == 'ACMINet':
                #     output, deeps = self.model(image)
                else:
                    raise RuntimeError("modalities error, chossen{}".format(self.args.modalities))

            output = torch.sigmoid(output)

            dice_loss = self.criterion1(output, target)
            bce_loss = self.criterion2(output, target)
            loss = dice_loss + bce_loss

            test_loss += loss.item()
            # print('Iter {} of {} Val loss {:.4f}'.format(iter + 1 , len(self.val_loader), (test_loss / (iter + 1))))
            target1 = target.cpu()
            target1 = target1.numpy()

            pred = (output > 0.5).float()
            pred = pred.long().cpu()
            pred = pred.numpy()
            hausdorff = metrics.hausdorff_95(pred[:, 0, :, :], target1[:, 0, :, :])
            hausdorff_sum += hausdorff
            self.evaluator1.update(pred[:, 0, None, :, :], target1[:, 0, None, :, :])

            iou = metrics.iou_score(pred[:, 0, :, :], target1[:, 0, :, :])
            iou_sum += iou


        # Fast test during the training
        dice = self.evaluator1.dice()
        hausdorff = hausdorff_sum / (iter + 1)
        iou = iou_sum / (iter + 1)

        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('val/dice', dice, epoch)
        self.writer.add_scalar('val/hausdorff', hausdorff, epoch)

        sensitivity = self.evaluator1.sensitivity()
        specificity = self.evaluator1.specificity()


        print('Validation:')
        #print('[Epoch: %d, numImages: %5d]' % (epoch, iter * self.args.batch_size + np.array(image).shape[0]))
        print('[Epoch: %d, numImages: %5d]' % (epoch, iter * self.args.batch_size ))
        print('dice: Lesion:{}'
              .format(dice).encode('utf-8'))
        print('hausdorff: Lesion:{}'
              .format(hausdorff).encode('utf-8'))
        print('sensitivity: Lesion:{}'
              .format(sensitivity).encode('utf-8'))
        print('specificity: Lesion:{}'
              .format(specificity).encode('utf-8'))
        print('IoU: Lesion:{}'
              .format(iou).encode('utf-8'))
        print('Loss: %.3f' % (test_loss / (iter + 1)))

        new_pred = dice
        if new_pred > self.best_pred:
            self.best_pred = new_pred

    def my_test(self, epoch):
        self.model.eval()
        test_loss = 0.0
        iter=0
        #image = sample.to(self.device)
        #target = sample.to(self.device)

        hausdorff_sum = 0.0
        iou_sum = 0.0


        for iter, sample in enumerate(self.test_loader):
            image = sample[0].to(self.device)
            target = sample[1].to(self.device)
            fn = sample[2]

            with torch.no_grad():
                if self.args.model == 'S2Net':
                    output = self.model(image)
                # elif self.args.model == 'UNet':
                #     output = self.model(image)
                # elif self.args.model == 'FFNet':
                #     output, output_m1, output_m2 = self.model(image)
                # elif self.args.model == 'ACMINet':
                #     output, deeps = self.model(image)
                else:
                    raise RuntimeError("modalities error, chossen{}".format(self.args.modalities))

            output = torch.sigmoid(output)
            dice_loss = self.criterion1(output, target)
            bce_loss = self.criterion2(output, target)
            loss = dice_loss + bce_loss
            test_loss += loss.item()
            # print('Iter {} of {} Test loss {:.4f}'.format(iter + 1 , len(self.val_loader), (test_loss / (iter + 1))))
            target1 = target.cpu()
            target1 = target1.numpy()

            pred = (output > 0.5).float()
            pred = pred.long().cpu()
            pred = pred.numpy()
            hausdorff = metrics.hausdorff_95(pred[:, 0, :, :], target1[:, 0, :, :])
            hausdorff_sum += hausdorff
            self.evaluator1.update(pred[:, 0, None, :, :], target1[:, 0, None, :, :])

            iou = metrics.iou_score(pred[:, 0, :, :], target1[:, 0, :, :])
            iou_sum += iou


        # Fast test during the training
        dice = self.evaluator1.dice()
        hausdorff = hausdorff_sum / (iter + 1)
        iou = iou_sum / (iter + 1)


        self.writer.add_scalar('test/total_loss_epoch', test_loss, epoch)
        self.writer.add_scalar('test/dice', dice, epoch)
        self.writer.add_scalar('test/hausdorff', hausdorff, epoch)

        sensitivity = self.evaluator1.sensitivity()
        specificity = self.evaluator1.specificity()


        print('Test:')
        #print('[Epoch: %d, numImages: %5d]' % (epoch, iter * self.args.batch_size + np.array(image).shape[0]))
        print('[Epoch: %d, numImages: %5d]' % (epoch, iter * self.args.batch_size ))
        print('dice: Lesion:{}'
              .format(dice).encode('utf-8'))
        print('hausdorff: Lesion:{}'
              .format(hausdorff).encode('utf-8'))
        print('sensitivity: Lesion:{}'
              .format(sensitivity).encode('utf-8'))
        print('specificity: Lesion:{}'
              .format(specificity).encode('utf-8'))
        print('IoU: Lesion:{}'
              .format(iou).encode('utf-8'))
        print('Loss: %.3f' % (test_loss / (iter + 1)))

        new_pred = dice
        if new_pred > self.best_pred:
            self.best_pred = new_pred