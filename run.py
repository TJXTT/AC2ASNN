import argparse
import os
import shutil
import time
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from tensorboardX import SummaryWriter
from block import *
from module import *
from networks import *


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='SNN2ANN Training')
parser.add_argument('--dataset', default='CIFAR100', type=str, help='dataset')
parser.add_argument('--data-path', default='../datasets/data_CIFAR100', type=str, help='data path')
parser.add_argument('--class-nums', default=100, type=int, help='class number')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet')
parser.add_argument('--time-steps', default=5, type=int)
parser.add_argument('--spike-unit', default='ReSU', type=str)
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N', help='mini-batch size (default: 512)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--kaiming-norm', default=False, type=bool, help='use kaiming normalization')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--checkpoint-path', default='./checkpoints', type=str, help='data path')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-load', default='', type=str, metavar='PATH',
                    help='path to training mask (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', default='', metavar='PATH',
                    help='use pre-trained model')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')

best_prec1 = 0
step_change_arr = [100, 150, 175]

tp1 = [];
tp5 = [];
tp1_sp = [];
tp5_sp = [];
ep = [];
lRate = [];
device_num = 8

tp1_tr = [];
tp5_tr = [];
tp1_tr_sp = [];
tp5_tr_sp = [];
losses_tr = [];
losses_tr_sp = [];
losses_eval = [];
losses_eval_sp = [];

def main():
    global args, best_prec1, batch_size, device_num
    args = parser.parse_args()

    time_steps = args.time_steps
    batch_size = args.batch_size
    arch = args.arch
    ckpt_path = args.checkpoint_path
    spike_unit = args.spike_unit
    kaiming_norm = args.kaiming_norm
    cls_nums = args.class_nums
    learning_rate = args.lr
    weight_decay = args.weight_decay
    data_path = args.data_path
    dataset = args.dataset.upper()
    writer = SummaryWriter('./summaries/'+arch+'_'+dataset+'_'+spike_unit+'_T='+str(time_steps))

    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    assert spike_unit in ['ReSU', 'STSU']

    # Model
    if arch.upper() == 'VGG':
        if dataset == 'TINY-IMAGENET':
            H = W = 64
            downsample_lst = [False, True, True, True, True]
        else:
            H = W = 32
            downsample_lst = [False, False, True, True, True]
        channel_lst = [64, 128, 256, 512, 512]
        model = VGG(H=H, W=W, C=3, num_classes=cls_nums, 
                     blocks=[VGGBlock_1,VGGBlock_1,VGGBlock_2,
                     VGGBlock_2,VGGBlock_2], 
                     channels=channel_lst, 
                     downsample=downsample_lst, 
                     T=time_steps, mapping_unit=spike_unit, 
                     kaiming_norm=kaiming_norm)
    else:
        if dataset == 'TINY-IMAGENET':
            H = W = 64
            stride_lst = [1, 1, 2, 2, 2]
        else:
            H = W = 32
            stride_lst = [1, 1, 1, 2, 2]
        channel_lst = [64, 64, 128, 256, 512]
        model =  ResNet(H=H, W=W, C=3, num_classes=cls_nums, 
                     strides=stride_lst, channels=channel_lst, 
                     T=time_steps, mapping_unit=spike_unit, 
                     kaiming_norm=kaiming_norm)

    print(model)
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(44)

    if device_num < 2:
        device = 0
        torch.cuda.set_device(device)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()
    
    # Data loading code
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.557, 0.549, 0.5534])
    if dataset == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_data = torchvision.datasets.CIFAR100(data_path, train=True, download=True, transform=transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        val_data = torchvision.datasets.CIFAR100(data_path, train=False, download=True, transform=transform_test)

    elif dataset == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_data = torchvision.datasets.CIFAR10(data_path, train=True, download=True, transform=transform_train)

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        val_data = torchvision.datasets.CIFAR10(data_path, train=False, download=True, transform=transform_test)

    elif dataset == 'TINY-IMAGENET':
        transform_train = transforms.Compose([
            #transforms.RandomResizedCrop(224),
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        train_path = os.path.join(data_path,'train')
        train_data =torchvision.datasets.ImageFolder(root=train_path, transform=transform_train)

        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
        val_path = os.path.join(data_path, 'val')
        val_data = torchvision.datasets.ImageFolder(root=val_path, transform=transform_test)

    else:
        raise Exception("Choose a dataset from CIFAR10, CIFAR100, or Tiny-ImageNet.")

    train_loader = torch.utils.data.DataLoader(train_data, 
                                                batch_size=args.batch_size, 
                                                shuffle=True, 
                                                num_workers=args.workers, 
                                                pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=False)

    criterion_en = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if os.path.exists(args.pretrained):
        print("=> loading pretrained model '{}'".format(args.pretrained))
        checkpoint = torch.load(args.pretrained)
        model.load_state_dict(checkpoint, strict=False)
    else:
        print("=> no pretrained model found at '{}'".format(args.pretrained))

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)
        ep.append(epoch)

        # train for one epoch
        start_time = time.time()
        loss_cnn, loss_snn, acc_cnn, acc_snn = train(train_loader, model, criterion_en, optimizer, epoch, time_steps=time_steps, writer=writer)
        # penalty = penalty * 2.0
        end_time = time.time()
        print('Train cost: {} sec/epoch'.format(end_time-start_time))
        # evaluate on validation set

        tr_loss = {'SNN-branch':loss_snn.avg,'CNN-branch':loss_cnn.avg}
        tr_acc = {'SNN-branch':acc_snn.avg,'CNN-branch':acc_cnn.avg}
        writer.add_scalars('Train Loss', tr_loss, epoch)
        writer.add_scalars('Train Top-1 Acc', tr_acc, epoch)
        start_time = time.time()
        prec1 = validate(val_loader, model, criterion_en, time_steps=time_steps, epoch=epoch, writer=writer)
        
        end_time = time.time()
        print('Eval cost: {} sec/epoch'.format(end_time-start_time))
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_file = dataset+'_'+arch.upper()+'_'+'T='+str(time_steps)+'_'+'epoch='+str(epoch)+'.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best, ckpt_path, save_file)

    for k in range(0, args.epochs - args.start_epoch):
        print('Epoch: [{0}/{1}]\t'
              'LR:{2}\t'
              'Prec@1 {top1:.3f} \t'
              'Prec@5 {top5:.3f} \t'
              'En_Loss_Eval {losses_en_eval: .4f} \t'
              'Prec@1_tr {top1_tr:.3f} \t'
              'Prec@5_tr {top5_tr:.3f} \t'
              'En_Loss_train {losses_en: .4f}'.format(
            ep[k], args.epochs, lRate[k], top1=tp1[k], top5=tp5[k], losses_en_eval=losses_eval[k], top1_tr=tp1_tr[k],
            top5_tr=tp5_tr[k], losses_en=losses_tr[k]))
    writer.close()



def train(train_loader, model, criterion_en, optimizer, epoch, time_steps, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1_tr = AverageMeter()
    top5_tr = AverageMeter()
    losses_en = AverageMeter()
    top1_tr_sp = AverageMeter()
    top5_tr_sp = AverageMeter()
    losses_en_sp = AverageMeter()
    top1_tr_sps = AverageMeter()
    top5_tr_sps = AverageMeter()
    losses_en_sps = AverageMeter()
    # switch to train mode
    model.train()
    for module in model.modules():
        module.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        data_time.update(time.time() - end)
        target = target.cuda()
        labels = Variable(target.cuda())
        if device_num < 2:
            input_var = Variable(input.cuda())
        else:
            input_var = torch.autograd.Variable(input.cuda())
        optimizer.zero_grad()  # Clear gradients w.r.t. parameters

        output, output_sp = model(input_var, steps=time_steps, epoch=epoch, training=True)
        boosting = False
        targetN = output.data.clone().zero_().cuda()
        targetN.scatter_(1, target.unsqueeze(1), 1)
        targetN = Variable(targetN.type(torch.cuda.FloatTensor))


        loss_en = criterion_en(output, labels).cuda()
        loss_sp = criterion_en(output_sp, labels).cuda()
        loss = loss_en
        loss.backward(retain_graph=False)
        optimizer.step()

        total_loss = loss
 
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))

        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        prec1_tr, prec5_tr = accuracy(output.data, target, topk=(1, 5))
        losses_en.update(loss_en.item(), input.size(0))
        top1_tr.update(prec1_tr.item(), input.size(0))
        top5_tr.update(prec5_tr.item(), input.size(0))
        prec1_tr_sp, prec5_tr_sp = accuracy(output_sp.data, target, topk=(1, 5))
        top1_tr_sp.update(prec1_tr_sp.item(), input.size(0))
        top5_tr_sp.update(prec5_tr_sp.item(), input.size(0))
        losses_en_sp.update(loss_sp.item(), input.size(0))
   
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 20 == 0:
            print('iter: {}, cnn_loss: {}, spike loss: {}'.format(i, loss_en.item(), loss_sp.item()))

    print('Epoch: [{0}] Prec@1 {top1_tr.avg:.3f} Prec@5 {top5_tr.avg:.3f} Entropy_Loss {loss_en.avg:.4f}'
          .format(epoch, top1_tr=top1_tr, top5_tr=top5_tr, loss_en=losses_en))
    print('SNN: Epoch: [{0}] Prec@1 {top1_tr.avg:.3f} Prec@5 {top5_tr.avg:.3f} Entropy_Loss {loss_en.avg:.4f}'
          .format(epoch, top1_tr=top1_tr_sp, top5_tr=top5_tr_sp, loss_en=losses_en_sp))

    losses_tr.append(losses_en.avg)
    tp1_tr.append(top1_tr.avg)
    tp5_tr.append(top5_tr.avg)
    return losses_en, losses_en_sp, top1_tr, top1_tr_sp


def validate(val_loader, model, criterion_en, time_steps, epoch, writer):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    losses_en_eval = AverageMeter()

    losses_cnn = AverageMeter()
    top1_cnn = AverageMeter()
    top5_cnn = AverageMeter()
    losses_en_eval_cnn = AverageMeter()

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()

    end = time.time()

    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        labels = Variable(target.cuda())
        target = target.cuda()
        if device_num < 2:
            input_var = Variable(input.cuda())
        else:
            input_var = torch.autograd.Variable(input.cuda())
        with torch.no_grad():
            output_cnn, output = model(input=input_var, steps=time_steps, epoch=epoch, training=False)
        targetN = output.data.clone().zero_().cuda()
        targetN.scatter_(1, target.unsqueeze(1), 1)
        targetN = Variable(targetN.type(torch.cuda.FloatTensor))

        loss_en = criterion_en(output, labels).cuda()
        loss_en_cnn = criterion_en(output_cnn, labels).cuda()


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        prec1_cnn, prec5_cnn = accuracy(output_cnn.data, target, topk=(1, 5))

        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        losses_en_eval.update(loss_en.item(), input.size(0))

        top1_cnn.update(prec1_cnn.item(), input.size(0))
        top5_cnn.update(prec5_cnn.item(), input.size(0))
        losses_en_eval_cnn.update(loss_en_cnn.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
    val_sq_loss = {'SNN-branch':losses.avg, 'CNN-branch':losses_cnn.avg}
    val_loss = {'SNN-branch':losses_en_eval.avg, 'CNN-branch':losses_en_eval_cnn.avg}
    val_acc = {'SNN-branch':top1.avg, 'CNN-branch':top1_cnn.avg}
    writer.add_scalars('Val Loss (CE)', val_loss, epoch)
    writer.add_scalars('Val Loss (MSE)', val_sq_loss, epoch)
    writer.add_scalars('Val Top-1 Acc', val_acc, epoch)
    vth_dict = {}
    for i, (name, param) in enumerate(model.named_parameters()):
        if 'Vth' in name:
            Vth_avg = torch.mean(param.data)
            vth_dict[name] = Vth_avg
    writer.add_scalars('Change of Vth', vth_dict, epoch)
    print('CNN Test: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Entropy_Loss {losses_en_eval.avg:.4f}'
          .format(top1=top1_cnn, top5=top5_cnn, losses_en_eval=losses_en_eval_cnn))

    print('SNN Test: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Entropy_Loss {losses_en_eval.avg:.4f}'
          .format(top1=top1, top5=top5, losses_en_eval=losses_en_eval))

    tp1.append(top1.avg)
    tp5.append(top5.avg)

    losses_eval.append(losses_en_eval.avg)


    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):

    for param_group in optimizer.param_groups:
        if epoch in step_change_arr:
            param_group['lr'] = param_group['lr']*0.1
    lRate.append(param_group['lr'])

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state, is_best, ckpt_path, filename='model.pth.tar'):
    save_path1 = os.path.join(ckpt_path, filename)
    torch.save(state, save_path1)
    if is_best:
        save_path2 = os.path.join(ckpt_path, 'model_best.pth.tar')
        shutil.copyfile(save_path1, save_path2)


if __name__ == '__main__':
    main()
