import argparse
import datetime
import os
import random
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.nn.parallel
import torch.optim
import torch.utils
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dataset import PromptTrainDataset, DenoiseTestDataset, DerainDehazeDataset

from models.network import Model

from utils.scheduler import *
from utils.metrics import *

parser = argparse.ArgumentParser()
parser.add_argument("--start-epoch", type=int, default=1, help="epoch to start training from")
parser.add_argument("--epochs", type=int, default=150, help="number of epochs of training")
parser.add_argument("--batch-size", type=int, default=64, help="size of the batches")

parser.add_argument("-j", "--workers", type=int, default=6, help="number of cpu threads to use during batch generation")
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

# Dataset Option
parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50', 'derain', 'dehaze', 'deblur', 'enhance'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--patch_size', type=int, default=128, help='patchsize of input.')

parser.add_argument('--data_file_dir', type=str, default='./data_dir/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='/home/jnu/Project/dataset/IR/Train/Denoise/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='/home/jnu/Project/dataset/IR/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='/home/jnu/Project/dataset/IR/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--gopro_dir', type=str, default='/home/jnu/Project/dataset/Motion_Deblurring/GOPRO/train',
                    help='where clean images of denoising saves.')
parser.add_argument('--enhance_dir', type=str, default='/home/jnu/Project/dataset/LowLight/LOLv1/our485',
                    help='where clean images of denoising saves.')

parser.add_argument('--output_path', type=str, default="./saved_models", help='output save path')

# Test
parser.add_argument('--denoise_path', type=str, default="/home/jnu/Project/dataset/IR/Test/denoise/", help='save path of test noisy images')
parser.add_argument('--derain_path', type=str, default="/home/jnu/Project/dataset/IR/Test/derain/", help='save path of test raining images')
parser.add_argument('--dehaze_path', type=str, default="/home/jnu/Project/dataset/IR/Test/dehaze/", help='save path of test hazy images')
parser.add_argument('--gopro_path', type=str, default="/home/jnu/Project/dataset/Motion_Deblurring/GOPRO/test", help='save path of test hazy images')
parser.add_argument('--enhance_path', type=str, default='/home/jnu/Project/dataset/LowLight/LOLv1/eval15', help='save path of test hazy images')
parser.add_argument('--mode', type=int, default=0,
                    help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

# model option
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--print-freq', type=int, default=1)
parser.add_argument('--eval-freq', type=int, help='Online evaluation frequency in global steps', default=1000)
parser.add_argument('--seed', type=int, default=None, help='random seed (default: None)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', help="model_args.resume")
parser.add_argument('--evaluate', '-e', default=False, action='store_true')
parser.add_argument('--tsne', '-t', default=False, action='store_true')

# Distributed
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    # global best_acc1
    args.gpu = gpu
    summary = SummaryWriter(log_dir='./runs/')

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create models
    model = Model()

    n_params = sum(p.numel() for p in model.parameters()) / 1_000_000
    learn_n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000
    print("[Total Params]: ", n_params)
    print("[Learning Params]: ", learn_n_params)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel w`ill divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()
    cudnn.benchmark = True

    # Optimizer
    criterion = nn.L1Loss().cuda(args.gpu)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.epochs,
                                                           eta_min=1e-6)
    train_dataset = PromptTrainDataset(args=args)

    denoise_splits = ["bsd68/"]
    derain_splits = ["Rain100L/"]
    denoise_tests = []
    derain_tests = []
    for i in denoise_splits:
        args.denoise_path = os.path.join(args.denoise_path, i)
        denoise_testset = DenoiseTestDataset(args)
        # denoise_tests.append(denoise_testset)

    for name in derain_splits:
        args.derain_path = os.path.join(args.derain_path, name)
        derain_set = DerainDehazeDataset(args, addnoise=False, sigma=25)

    dehaze_base_path = args.dehaze_path
    # name = derain_splits[0]
    args.dehaze_path = os.path.join(dehaze_base_path)
    dehaze_set = DerainDehazeDataset(args, addnoise=False, sigma=25, task='dehaze')

    denoise_testset.set_sigma(sigma=25)
    derain_set.set_dataset('derain')
    dehaze_set.set_dataset('dehaze')

    # Sampler
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_denoise_sampler = torch.utils.data.distributed.DistributedSampler(denoise_testset, shuffle=False, drop_last=True)
        test_derain_sampler = torch.utils.data.distributed.DistributedSampler(derain_set, shuffle=False, drop_last=True)
        test_dehaze_sampler = torch.utils.data.distributed.DistributedSampler(dehaze_set, shuffle=False, drop_last=True)
    else:
        train_sampler = None
        test_denoise_sampler = None
        test_derain_sampler = None
        test_dehaze_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None), drop_last=True,
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    denoise_test_loader = torch.utils.data.DataLoader(denoise_testset, batch_size=1,
                                                      shuffle=False,
                                                      num_workers=1, pin_memory=True, sampler=test_denoise_sampler)
    derain_test_loader = torch.utils.data.DataLoader(derain_set, batch_size=1,
                                                     shuffle=False,
                                                     num_workers=1, pin_memory=True, sampler=test_derain_sampler)
    dehaze_test_loader = torch.utils.data.DataLoader(dehaze_set, batch_size=1,
                                                     shuffle=False,
                                                     num_workers=1, pin_memory=True, sampler=test_dehaze_sampler)

    # Resume
    best_ssim = 0
    best_psnr = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_psnr = checkpoint['PSNR']
            best_ssim = checkpoint['SSIM']

            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print(f'PSNR: {best_psnr} | SSIM: {best_ssim}')
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        epoch = 1
        model.eval()
        print("--[validate denoise[sig=25]]--")
        denoise_SSIM, denoise_PSNR = test(model,
                                          denoise_test_loader,
                                          epoch, summary, args, ngpus_per_node)
        print("--[validate derain]--")
        derain_SSIM, derain_PSNR = test(model,
                                        derain_test_loader,
                                        epoch, summary, args, ngpus_per_node)  # 89
        print("--[validate dehaze]--")
        dehaze_SSIM, dehaze_PSNR = test(model,
                                        dehaze_test_loader,
                                        epoch, summary, args, ngpus_per_node)
        print(f"PSNR - Denoise: {denoise_PSNR: .4f} | Derain: {derain_PSNR: .4f} | Dehaze: {dehaze_PSNR: .4f} |")
        print(f"SSIM - Denoise: {dehaze_SSIM: .4f} | Derain: {derain_SSIM: .4f} | Dehaze: {dehaze_SSIM: .4f} |")
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(model,
              train_loader,
              criterion,
              optimizer, scheduler,
              epoch, args, summary)

        print("--[validate denoise[sig=25]]--")
        denoise_SSIM, denoise_PSNR = test(model,
                                          denoise_test_loader,
                                          epoch, summary, args, ngpus_per_node)
        print("--[validate derain]--")
        derain_SSIM, derain_PSNR = test(model,
                                        derain_test_loader,
                                        epoch, summary, args, ngpus_per_node)
        print("--[validate dehaze]--")
        dehaze_SSIM, dehaze_PSNR = test(model,
                                        dehaze_test_loader,
                                        epoch, summary, args, ngpus_per_node)

        print(f"PSNR - Denoise: {denoise_PSNR: .4f} | Derain: {derain_PSNR: .4f} | Dehaze: {dehaze_PSNR: .4f} |")
        print(f"SSIM - Denoise: {denoise_SSIM: .4f} | Derain: {derain_SSIM: .4f} | Dehaze: {dehaze_SSIM: .4f} |")
        PSNR = denoise_PSNR + derain_PSNR + dehaze_PSNR
        SSIM = denoise_SSIM + derain_SSIM + dehaze_SSIM

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            print(f'Epoch: {epoch} | SSIM: {SSIM:.4f} | PSNR: {PSNR:.4f}')
            if PSNR > best_psnr:
                print("[Found better validated model]")
                best_psnr = PSNR
                best_ssim = SSIM

                if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                            and args.rank % ngpus_per_node == 0):
                    torch.save({
                        'epoch': epoch + 1,
                        'PSNR': best_psnr,
                        'SSIM': best_ssim,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict()
                    }, "saved_models/base/checkpoint_%d.pth" % epoch)


def train(model,
          train_loader,
          criterion,
          optimizer, scheduler,
          epoch, args, summary):
    model.train()

    end = time.time()
    for i, (id, image, target) in enumerate(train_loader):
        # Cosine Scheulder
        niter = (epoch - 1) * len(train_loader) + i
        for param_group in optimizer.param_groups:
            lr = param_group["lr"]
        image = image.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        out = model(image)
        loss_rec = criterion(out, target)

        target_freq = torch.fft.rfft2(target, dim=(-2, -1), norm='ortho')
        target_freq = torch.stack((target_freq.real, target_freq.imag), dim=-1)
        pred_freq = torch.fft.rfft2(out, dim=(-2, -1), norm='ortho')
        pred_freq = torch.stack((pred_freq.real, pred_freq.imag), dim=-1)

        loss_fft = criterion(pred_freq, target_freq)
        loss_cr = 0
        loss = loss_rec + loss_cr + loss_fft

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.gpu == 0:
            summary.add_scalar('Train/loss', loss.item(), niter)
            summary.add_scalar('Train/rec', loss_rec, niter)
            summary.add_scalar('Train/CR', loss_cr, niter)
            summary.add_scalar('Train/lr', lr, niter)

        if i % args.print_freq == 0 or (epoch == 1 and i < 100):
            print(f"Epoch [{epoch}][{i}/{len(train_loader)}] | Loss: {loss: .4f} | loss_rec: {loss_rec: .4f} | loss_cr: {loss_cr: .4f} | loss_fft: {loss_fft: .4f} | Lr: {lr: .4f} |")
    scheduler.step()
    elapse = datetime.timedelta(seconds=time.time() - end)
    print(f"걸린 시간: ", elapse)


def test(model,
         test_loader,
         epoch,
         summary, args, ngpus_per_node):
    eval_measures = torch.zeros(3).cuda(args.gpu)

    model.eval()
    total_ssim = 0
    total_psnr = 0

    # factor = 16
    with torch.no_grad():
        for i, (id, image, target) in tqdm(enumerate(test_loader)):
            image = image.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)
            out = model(image)
            now_ssim = ssim(out, target).item()
            total_ssim += now_ssim

            now_psnr = psnr(out, target)
            total_psnr += now_psnr
            eval_measures[0] += torch.tensor(now_ssim)
            eval_measures[1] += torch.tensor(now_psnr)
            eval_measures[2] += 1

        if args.multiprocessing_distributed:
            group = dist.new_group([i for i in range(ngpus_per_node)])
            dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

        if not args.multiprocessing_distributed or args.gpu == 0:
            eval_measures = eval_measures.cpu()
            cnt = eval_measures[-1].item()
            eval_measures /= cnt

        summary.add_scalar('Valid/psnr', eval_measures[1], epoch)
        summary.add_scalar('Valid/ssim', eval_measures[0], epoch)
        return eval_measures[0], eval_measures[1]


if __name__ == "__main__":
    main()

