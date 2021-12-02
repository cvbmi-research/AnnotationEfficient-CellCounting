import sys, os, warnings
from utils import save_checkpoint
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
import argparse, json, cv2, time
import VGG_dataset_whole_labeled as labeled_dataset
import VGG_dataset_whole_unlabeled as unlabeled_dataset
import VGG_dataset_whole_val as val_dataset
from VGG_model import *
from torchsummary import summary
from PIL import Image

parser = argparse.ArgumentParser(description='EncoDeco_MSE_')
parser.add_argument('labeled_train_json', metavar='TRAIN', help='path to labeled_train json')
parser.add_argument('unlabeled_train_json', metavar='TRAIN', help='path to unlabeled train json')
parser.add_argument('test_json', metavar='TEST', help='path to test json')
parser.add_argument('--pre_g1', '-p_g1', metavar='PRETRAINED_NET', default=None, type=str, help='path to the pretrained EncoDeco model1')
parser.add_argument('--pre_g2', '-p_g2', metavar='PRETRAINED_NET', default=None, type=str, help='path to the pretrained EncoDeco model2')
parser.add_argument('--pre_g3', '-p_g3', metavar='PRETRAINED_NET', default=None, type=str, help='path to the pretrained EncoDeco model3')
parser.add_argument('gpu',metavar='GPU', type=str, help='GPU id to use.')
parser.add_argument('task',metavar='TASK', type=str, help='task id to use.')

def main():
    global args, best_prec1, best_prec2, best_prec3

    best_prec1 = 1e6
    best_prec2 = 1e6
    best_prec3 = 1e6

    args = parser.parse_args()
    args.original_lr = 5e-5
    args.lr = 5e-5
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 100
    args.steps         = [-1,300,400,450] # adjust learning rate
    args.scales        = [1, 0.1, 0.01, 0.001] # adjust learning rate
    args.workers = 8
    args.seed = 0 
    args.print_freq = 5
    args.init_round = 50
    args.save_inter = 20    
    
    img_shape = (256, 256) # for VGG datset


    args.arch = args.task + 'EnDe50_sa_mse_' 
    with open(args.labeled_train_json, 'r') as outfile:        
        labeled_train_list = json.load(outfile) # labeled data
    with open(args.unlabeled_train_json, 'r') as outfile:        
        unlabeled_train_list = json.load(outfile) # unlabeled data
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)

    os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.empty_cache()

    model1 = Generator().cuda()
    model2 = Generator().cuda()
    model3 = Generator().cuda()


    criterion11 = torch.nn.MSELoss(size_average=False, reduction='none').cuda() # for counting
    criterion12 = torch.nn.CrossEntropyLoss(size_average=False, reduction='none').cuda()  # for clustering centers
    criterion13 = torch.nn.KLDivLoss(size_average=False, reduction='none').cuda() # for reconstruction
    criterion14 = torch.nn.MSELoss(size_average=False, reduction='none').cuda() # for reconstruction
    criterion15 = torch.nn.L1Loss(size_average=False, reduction='none').cuda() # for reconstruction
    
    criterion21 = torch.nn.MSELoss(size_average=False, reduction='none').cuda() 
    criterion22 = torch.nn.CrossEntropyLoss(size_average=False, reduction='none').cuda()
    criterion23 = torch.nn.KLDivLoss(size_average=False, reduction='none').cuda()
    criterion24 = torch.nn.MSELoss(size_average=False, reduction='none').cuda()
    criterion25 = torch.nn.L1Loss(size_average=False, reduction='none').cuda()
    
    criterion31 = torch.nn.MSELoss(size_average=False, reduction='none').cuda()
    criterion32 = torch.nn.CrossEntropyLoss(size_average=False, reduction='none').cuda()
    criterion33 = torch.nn.KLDivLoss(size_average=False, reduction='none').cuda()
    criterion34 = torch.nn.MSELoss(size_average=False, reduction='none').cuda()
    criterion35 = torch.nn.L1Loss(size_average=False, reduction='none').cuda()
    
    g_opti1 = torch.optim.AdamW(model1.parameters(), lr = args.lr, weight_decay=args.decay)
    g_opti2 = torch.optim.AdamW(model2.parameters(), lr = args.lr, weight_decay=args.decay)
    g_opti3 = torch.optim.AdamW(model3.parameters(), lr = args.lr, weight_decay=args.decay)



    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(g_opti1, epoch)
        adjust_learning_rate(g_opti2, epoch)
        adjust_learning_rate(g_opti3, epoch)

        train(labeled_train_list, model1, criterion11, criterion12, criterion13, criterion14, criterion15, g_opti1, epoch)  
        train(labeled_train_list, model2, criterion21, criterion22, criterion23, criterion24, criterion25, g_opti2, epoch)  
        train(labeled_train_list, model3, criterion31, criterion32, criterion33, criterion34, criterion35, g_opti3, epoch)  
        
        prec1 = validate(val_list, model1)
        prec2 = validate(val_list, model2)
        prec3 = validate(val_list, model3)

        is_best1 = prec1 < best_prec1
        is_best2 = prec2 < best_prec2
        is_best3 = prec3 < best_prec3

        best_prec1 = min(prec1, best_prec1)
        best_prec2 = min(prec2, best_prec2)
        best_prec3 = min(prec3, best_prec3)
        
        line1 = '*** VGG Best(m1) _MAE_ {mae:.3f} ***'.format(mae=best_prec1)
        line2 = '*** VGG Best(m2) _MAE_ {mae:.3f} ***'.format(mae=best_prec2)
        line3 = '*** VGG Best(m3) _MAE_ {mae:.3f} ***'.format(mae=best_prec3)

        with open('logs/Log-{}_{}.txt'.format(time_stp, args.arch), 'a+') as flog:
            print(line1)
            print(line2)
            print(line3)
            flog.write('{}\n'.format(line1))
            flog.write('{}\n'.format(line2))
            flog.write('{}\n'.format(line3))
        
        # save best val models:
        if is_best1:
            save_checkpoint({
                'epoch': epoch + 1,
                'g_arch': args.pre_g1,
                'g_state_dict': model1.state_dict(),
                'best_prec1': best_prec1,
                'g_optimizer' : g_opti1.state_dict()
                }, is_best1, args.task + "m1_")
        
        if is_best2:
            save_checkpoint({
                'epoch': epoch + 1,
                'g_arch': args.pre_g2,
                'g_state_dict': model2.state_dict(),
                'best_prec1': best_prec2,
                'g_optimizer' : g_opti2.state_dict()
                }, is_best2, args.task + "m2_")

        if is_best3:
            save_checkpoint({
                'epoch': epoch + 1,
                'g_arch': args.pre_g3,
                'g_state_dict': model3.state_dict(),
                'best_prec1': best_prec3,
                'g_optimizer' : g_opti3.state_dict()
                }, is_best3, args.task + "m3_")


        if (epoch + 1) % args.save_inter == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'g_arch': args.pre_g1,
                'g_state_dict': model1.state_dict(),
                'best_prec1': best_prec1,
                'g_optimizer' : g_opti1.state_dict()
                }, False, args.task + "m1_" + str(epoch+1) + "_")

            save_checkpoint({
                'epoch': epoch + 1,
                'g_arch': args.pre_g2,
                'g_state_dict': model2.state_dict(),
                'best_prec1': best_prec2,
                'g_optimizer' : g_opti2.state_dict()
            }, False, args.task + "m2_" + str(epoch+1) + "_")

            save_checkpoint({
                'epoch': epoch + 1,
                'g_arch': args.pre_g3,
                'g_state_dict': model3.state_dict(),
                'best_prec1': best_prec3,
                'g_optimizer' : g_opti3.state_dict()
            }, False, args.task + "m3_" + str(epoch+1) + "_")



def train(train_list, model, criterion1, criterion2, criterion3, criterion4, criterion5, g_opti, epoch): 
    # rest all the parameters
    g_losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    
    if epoch > args.init_round:
        train_loader = torch.utils.data.DataLoader(
            unlabeled_dataset.listDataset2(train_list,
                        shuffle=True,
                        transform=transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),
                        train=True),
            batch_size=args.batch_size,
            num_workers=8)
        print('Unsupervised_ONLY Epoch %d, processed %d samples, lr %.10f, dsf_%s' % (epoch, epoch * len(train_loader.dataset), args.lr, args.task.split('dsf')[-1][:3]))    
        model.train()
        end = time.time()
        
        for i, (img, img2) in enumerate(train_loader):
            data_time.update(time.time() - end)
            img = img.cuda()
            img = Variable(img)

            g_opti.zero_grad()
            g_dMap, cluster_1 = model(img)
            g_dMap = g_dMap.squeeze(0).squeeze(0)
            g_dMap = g_dMap
            new_input = torch.stack((g_dMap, g_dMap,g_dMap)).unsqueeze(0)
            g_dMap2, cluster_2 = model(new_input)
            input1 = torch.log_softmax(cluster_1/0.1, dim=1)
            input2 = torch.softmax(cluster_2/0.1, dim=1)
            clu2_loss = criterion3(input1, input2)
            l1_loss = criterion4(g_dMap2, g_dMap)
            
            par3 = 1.0
            g_loss = par3*clu2_loss + l1_loss
            g_losses.update(g_loss.item(), img.size(0))
            g_loss.backward()
            g_opti.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                with open('logs/Log-{}_{}.txt'.format(time_stp, args.arch), 'a+') as flog:
                    line = 'Unsupervised_VGG Epoch: [{0}][{1}/{2}]  ' \
                        'Loss: {g_loss.val:.3f} ({g_loss.avg:.3f})  ' 'ClusterLoss: {recons_loss:.3f}  ' 'L1Loss: {l1_loss:.3f}'.format(epoch, i, len(train_loader), g_loss=g_losses, recons_loss=clu2_loss, l1_loss=l1_loss)
                    print(line)
                    flog.write('{}\n'.format(line))   


    # supervised-training
    train_loader = torch.utils.data.DataLoader(
        labeled_dataset.listDataset1(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]),
                       train=True),
        batch_size=args.batch_size,
        num_workers=8)
    print('Supervised Epoch %d, processed %d samples, lr %.10f, dsf_%s' % (epoch, epoch * len(train_loader.dataset), args.lr, args.task.split('dsf')[-1][:3]))
    model.train()
    end = time.time()

    for i, (img, target, cluster_center) in enumerate(train_loader):
        data_time.update(time.time() - end)
        img = img.cuda()
        img = Variable(img)
        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)
        cluster_center = cluster_center.type(torch.LongTensor).cuda()
        target_cluster = Variable(cluster_center)

        g_opti.zero_grad()
        g_dMap, g_cluster = model(img)
        g_cluster = g_cluster/0.1
        cluster_cp = g_cluster
       
        mse_loss = criterion1(g_dMap, target)
        cluster_loss = criterion2(g_cluster, target_cluster)

        empty_bg = torch.zeros_like(g_dMap).cuda()
        sparse_loss = criterion5(g_dMap, empty_bg)        
        
        par2 = 0.001
        par3 = 0.0001
        g_loss = mse_loss + par2*cluster_loss + par3*sparse_loss # additional sparse loss, not necessary
        g_losses.update(g_loss.item(), img.size(0))
        g_loss.backward()
        g_opti.step()

        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            with open('logs/Log-{}_{}.txt'.format(time_stp, args.arch), 'a+') as flog:
                line = 'Supervised_VGG Epoch: [{0}][{1}/{2}]  ' \
                       'Loss: {g_loss.val:.3f} ({g_loss.avg:.3f})  ' 'MSELoss: {mse_loss:.3f}  ' 'ClusterLoss: {cluster_loss:.3f} '.format(epoch, i, len(train_loader), g_loss=g_losses, mse_loss=mse_loss, cluster_loss=cluster_loss)
                print(line)
                flog.write('{}\n'.format(line))


def validate(val_list, model):
    print ('|||---------begin test---------|||\n')
    test_loader = torch.utils.data.DataLoader(
        val_dataset.listDataset3(val_list,
                    shuffle=False,
                    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])]), 
                    train=False), 
        batch_size=args.batch_size,
        num_workers=8)

    model.eval()
    val_mae = 0.0
    val_dist = 0.0

    for i,(img, target) in enumerate(test_loader):
        img = img.cuda()
        img = Variable(img)
        v_dMap, _, = model(img)
        d_sum = v_dMap.data.cpu().clone().detach().numpy() 
        val_mae += abs(d_sum.sum() - target.sum()) 

    val_mae = val_mae/len(test_loader)
    line = '**** VGG MAE(avg) {val_mae:.3f} **** '.format(val_mae=val_mae)
    with open('logs/Log-{}_{}.txt'.format(time_stp, args.arch), 'a+') as flog:
        print(line)
        flog.write('{}\n'.format(line))
    return val_mae


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 0.1 every 25 epochs"""
    args.lr = args.original_lr
    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

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

if __name__ == '__main__':
    time_stp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    main()