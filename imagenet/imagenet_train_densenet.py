import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import math
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
import densenet as densenet

# used for logging to TensorBoard
# from tensorboard_logger import configure, log_value

parser = argparse.ArgumentParser(description='PyTorch DenseNet Training')
parser.add_argument('--epochs', default=135, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=56*4, type=int,
                    help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--no-augment', dest='augment', action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='DenseNet_BC_100_12', type=str,
                    help='name of experiment')
parser.add_argument('--tensorboard',
                    help='Log progress to TensorBoard', action='store_true')
parser.set_defaults(augment=True)
#parser.set_defaults(resume='save/save_models/checkpoint_071.pth.tar')
best_prec1 = 0




def main():
    global args, best_prec1
    args = parser.parse_args()
    # if args.tensorboard: configure("runs/%s"%(args.name))

    # Data loading code
    traindir = os.path.join('../datasets', 'train')
    valdir = os.path.join('../datasets', 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_set = datasets.ImageFolder(traindir, transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

    val_set = datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ]))

    kwargs = {'num_workers': 8, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=64, shuffle=True, **kwargs)

    # create model
    # model = dn.DenseNet()
    model = densenet.DenseNet()

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    
    model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
    # optionally resume from a checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
    for epoch in range(args.start_epoch, args.epochs):
        ### Train for one epoch
        tr_prec1, tr_prec5, loss, lr, train_time = \
            train(train_loader, model, criterion, optimizer, epoch)

        ### Evaluate on validation set
        val_prec1, val_prec5, val_time = validate(val_loader, model, criterion, epoch)

        ### Remember best prec@1 and save checkpoint
        is_best = (100. - val_prec1) < best_prec1
        best_prec1 = max(100. - val_prec1, best_prec1)
        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        line = "epoch:%d,time:%.4f,val_pre1:%.4f,val_pre5:%.4f,tr_pre1:%.4f,tr_pre5:%.4f,tr_loss:%.4f,lr:%.4f\n" % (
            epoch, train_time + val_time, val_prec1, val_prec5, tr_prec1, tr_prec5, loss, lr)
        save_checkpoint({
            'epoch': epoch,
            # 'model': args.model,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best, model_filename, line)

        # with open('logs_imagenet.txt', 'a') as f:
        #    f.write(line)
        # print(line)
    print('Best accuracy: ', best_prec1)


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    learned_module_list = []

    total_time_start = time.time()
    ### Switch to train mode
    model.train()
    ### Find all learned convs to prepare for group lasso loss
    for m in model.modules():
        if m.__str__().startswith('Learned_Dw_Conv'):
            learned_module_list.append(m)
    running_lr = None
    first = -1
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # progress = float(epoch * len(train_loader) + i) / \
        #    (args.epochs * len(train_loader))
        # args.progress = progress
        ### Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method='cosine')
        if running_lr is None:
            running_lr = lr
        if first == -1:
            first = 1
            progress = epoch
        else:
            progress = epoch + 0.1
        ### Measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        ### Compute output
        output = model(input_var, progress)
        loss = criterion(output, target_var)
        ldw_loss = 0
        for m in learned_module_list:
            ldw_loss = ldw_loss + m.ldw_loss

        loss = loss + 1e-4 * ldw_loss

        ### Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        ### Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f}\t'  # ({batch_time.avg:.3f}) '
                  'Data {data_time.val:.3f}\t'  # ({data_time.avg:.3f}) '
                  'Loss {loss.val:.4f}\t'  # ({loss.avg:.4f}) '
                  'Prec@1 {top1.val:.3f}\t'  # ({top1.avg:.3f}) '
                  'Prec@5 {top5.val:.3f}\t'  # ({top5.avg:.3f})'
                  'lr {lr: .4f}'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=lr))
    total_time_end = time.time()
    return 100. - top1.avg, 100. - top5.avg, losses.avg, running_lr, total_time_end - total_time_start


def load_checkpoint(args):
    # model_dir = os.path.join(args.savedir, 'save_models')
    # latest_filename = os.path.join(model_dir, 'latest.txt')
    # if os.path.exists(latest_filename):
    #    with open(latest_filename, 'r') as fin:
    #        model_filename = fin.readlines()[0]
    # else:
    #    return None
    # print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(args)
    # print("=> loaded checkpoint '{}'".format(model_filename))
    return state


def validate(val_loader, model, criterion, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    ### Switch to evaluate mode
    model.eval()
    total_time_start = time.time()
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        ### Compute output
        output = model(input_var, epoch + 0.1)
        loss = criterion(output, target_var)

        ### Measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))

        ### Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
          .format(top1=top1, top5=top5))
    total_time_end = time.time()
    return 100. - top1.avg, 100. - top5.avg, total_time_end - total_time_start


def save_checkpoint(state, args, is_best, filename, result):
    # print(args)
    savedir = 'save'
    result_filename = os.path.join(savedir, 'imagenetlogs')
    model_dir = os.path.join(savedir, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(savedir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))
    with open(result_filename, 'a') as fout:
        fout.write(result)
    print(result)
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    # if args.no_save_model:
    #    shutil.move(model_filename, best_filename)
    if is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return


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
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='cosine'):
    if method == 'cosine':

        if epoch < 45:
            T_total = 90 * nBatch
            T_cur = (epoch % 90) * nBatch + batch
        else:
            T_total = (90) * nBatch
            T_cur = ((epoch-45) % 90) * nBatch + batch

        #T_total = 90 * nBatch
        #T_cur = (epoch % 90) * nBatch + batch

        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data in ['cifar10', 'cifar100']:
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = args.lr * (0.1 ** (epoch // 90))
    # lr=args.lr*(0.1 ** (epoch//180))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
