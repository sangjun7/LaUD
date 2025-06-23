import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import *
from model import *
from utils import *

#===============================================================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--nepoch', default=3000, type=int, help='Number of Training epochs')
parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate')
parser.add_argument('--batch', default=16, type=int, help='Train batch size')
parser.add_argument('--mag', default=2, type=int, help='Upscaling factor (choices: 2,4,8)')
parser.add_argument('--lr_schedule', default=200, type=int, help='Decay period of learning rate')
parser.add_argument('--decay_rate', default=0.5, type=float, help='Decay rate of learning rate')

parser.add_argument('--rcrop_sz', default=96, type=int, help='Target size for randomly cropping the input image')
parser.add_argument('--train_dir', default=None, help='Directory for the training dataset')
parser.add_argument('--test_dir', default=None, help='Directory for the test dataset')
parser.add_argument('--test_data', default='Set5', help='Test dataset')
parser.add_argument('--load_checkpoint', default=None, help='Path to the model checkpoint to load')

parser.add_argument('--save_dir', default='./Models', help='Directory to save the trained model')
parser.add_argument('--save_model_name', default='DRLN', help='Name for trained model')
parser.add_argument('--print_iter', default=500, type=int, help='Interval of iterations to print training information')
parser.add_argument('--save_epoch', default=100, type=int, help='Epoch interval for saving model')
parser.add_argument('--test_epoch', default=100, type=int, help='Epoch interval for running validation during training')
parser.add_argument('--last_save', action='store_true', help='Whether to save the trained model after the final epoch')

parser.add_argument('--workers', default=4, type=int, help='Number of workers')
parser.add_argument('--cuda', action='store_true', help='Whether to use CUDA for computation')
parser.add_argument('--ngpu', default=1, type=int, help='Number of GPUs to utilize')
parser.add_argument('--initial_gpu', default=0, type=int, help='Initial GPU device number')
parser.add_argument('--seed', default=1, type=int, help='Manual Seed Number')

args = parser.parse_args() 

#===============================================================================================================================
print('Random Seed: ', args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)

use_cuda = False
device = torch.device('cpu')
if torch.cuda.is_available() is False and args.cuda:
    print("WARNING: You don't have a CUDA device so this code will be run on CPU.")
elif torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device but you don't use the device this time.")
elif torch.cuda.is_available() and args.cuda:
    use_cuda = True
    device = torch.device('cuda:{}'.format(args.initial_gpu))
    torch.cuda.set_device(device)
    print ('Current cuda device ', torch.cuda.current_device())
    
#===============================================================================================================================
model = DRLN(scale=args.mag)
if args.load_checkpoint is not None:
    model.load_state_dict(torch.load(args.load_checkpoint))
if args.ngpu > 1:
    model = nn.DataParallel(model, device_ids=list(range(args.initial_gpu, args.initial_gpu + args.ngpu)))
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-8)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_schedule, gamma=args.decay_rate)
criterion = nn.L1Loss().to(device)

#===============================================================================================================================
trainset = train_dataset(mode='train', root_path=args.train_dir,
                         is_resize=False, resize_h=None, resize_w=None, is_rcrop=True, crop_h=args.rcrop_sz, crop_w=None, scale=args.mag,
                         is_rrot=True, rand_hori_flip=True, rand_vert_flip=True, rgb_shuffle=False, grayscale=False, norm=False)
trainloader = DataLoader(trainset, batch_size=args.batch, shuffle=True, num_workers=args.workers)
        
#===============================================================================================================================
model.train()
start_time = time.time()
for epoch in range(args.nepoch):
    loss_sum = 0
    for iteration, data in enumerate(trainloader):
        hr_img, lr_img = data[0].requires_grad_(requires_grad=False), data[1]
        if use_cuda:
            hr_img = hr_img.to(device)
            lr_img = lr_img.to(device)
        
        optimizer.zero_grad()
        sr = model(lr_img)
        
        loss = criterion(sr, hr_img)
            
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        
        if iteration == 0 or (iteration+1) % args.print_iter == 0:
            info = '[{}/{}] Epoch, {} Iteration ==> Time : {:.2f} '.format(epoch+1, args.nepoch, iteration+1, time.time()-start_time)
            info += 'Iter_SR_loss : {:.4f}'.format(loss.item())
            print(info)
        
    print('[{}/{}] Epoch, Avg_SR_loss : {:.4f}'.format(epoch+1, args.nepoch, loss_sum/len(trainloader)))
    
    if (epoch+1) % args.save_epoch == 0:
        if args.ngpu <= 1 or use_cuda is False:
            torch.save(model.state_dict(), '/'.join([args.save_dir, args.save_model_name+'_X{}_{}epoch.pt'.format(args.mag, epoch+1)]))
        elif use_cuda and args.ngpu > 1:
            torch.save(model.module.state_dict(), '/'.join([args.save_dir, args.save_model_name+'_X{}_{}epoch.pt'.format(args.mag, epoch+1)]))
        
    scheduler.step()
    
    if (epoch+1) % args.test_epoch == 0:
        testset = test_dataset(root_path=args.test_dir, type=args.test_data,
                               is_resize=False, resize_h=None, resize_w=None, is_rcrop=False, crop_h=None, crop_w=None, scale=args.mag, 
                               is_rrot=False, rand_hori_flip=False, rand_vert_flip=False, rgb_shuffle=False, grayscale=False, norm=False)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.workers)
        model.eval()
        test_time = time.time()
        with torch.no_grad():
            psnr_sum = 0 
            ssim_sum = 0
            for titeration, tdata in enumerate(testloader):
                thr_img, tlr_img = tdata[0].requires_grad_(requires_grad=False), tdata[1]
                if use_cuda:
                    thr_img = thr_img.to(device)
                    tlr_img = tlr_img.to(device)
                
                tsr = model(tlr_img)
                
                psnr_sum += cal_psnr(thr_img, tsr, crop_border=args.mag, minmax='0_1', clamp=True, gray_scale=True, ver='YCrCb_BT601')
                ssim_sum += cal_ssim(thr_img, tsr, crop_border=args.mag, minmax='0_1', filter_size=11, filter_sigma=1.5, clamp=True, grayscale=True, ver='YCrCb_BT601')
                
            tinfo = "Time : {:.2f} s, Average PSNR: {:.2f} dB, Average SSIM: {:.4f}".format(time.time()-test_time, psnr_sum/len(testloader),
                                                                                            ssim_sum/len(testloader))
            print(tinfo)
        model.train()
        
print('Train Complete\n')
if args.last_save:
    if args.ngpu <= 1 or use_cuda is False:
        torch.save(model.state_dict(), '/'.join([args.save_dir, args.save_model_name+'_X{}_{}epoch.pt'.format(args.mag, args.nepoch)]))
    elif use_cuda and args.ngpu > 1:
        torch.save(model.module.state_dict(), '/'.join([args.save_dir, args.save_model_name+'_X{}_{}epoch.pt'.format(args.mag, args.nepoch)]))
