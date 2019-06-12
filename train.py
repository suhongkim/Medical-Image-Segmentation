import sys
import os
from os.path import join
from optparse import OptionParser
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torchvision import transforms

from model import UNet
from dataloader import DataLoader


def train_net(net,
              epochs=5,
              data_dir='data/cells/',
              n_classes=2,
              lr=0.001, # 0.1
              val_percent=0.1,
              save_cp=True,
              gpu=False):

    print("Training Start!!!!")
    start_time = time.time()

    loader = DataLoader(data_dir)

    N_train = loader.n_train()
    criterion = nn.CrossEntropyLoss() # for comparision
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.99,
                          weight_decay=0.0005)

    max_acc = 0
    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))

        net.train()
        loader.setMode('train')

        epoch_loss = 0
        for i, (img, label) in enumerate(loader):
            shape = img.shape
            label = label - 1
            # todo: create image tensor: (N,C,H,W) - (batch size=1,channels=1,height,width)
            image_t = torch.from_numpy(img.reshape(1, 1, shape[0], shape[1])).float()
            label_t = torch.from_numpy(label).float()
            # todo: load image tensor to gpu
            if gpu:
                image_t = image_t.cuda()
                label_t = label_t.cuda()

            # todo: get prediction and getLoss()
            pred_t = net.forward(image_t)
            loss = getLoss(pred_t, label) # don't use torch for label (Choose func use numpy) 
            epoch_loss += loss.item()
            #print("Torch CrossEntropyLoss: ", criterion(pred_t, label_t.view(1, label_t.shape[0], label_t.shape[1]).long()))
            #print('Training sample %d / %d - Loss: %.6f' % (i+1, N_train, loss.item()))

            # optimize weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        torch.save(net.state_dict(), join('/home/suhongk/sfuhome/','data') + '/CP%d.pth' % (epoch + 1))
#         torch.save(net.state_dict(), join(data_dir, 'checkpoints') + '/CP%d.pth' % ((epoch + 1))
        print('Checkpoint %d saved !' % (epoch + 1))

        print('Epoch %d finished! - Loss: %.6f' % (epoch+1, epoch_loss / i))
        pred_sm = softmax(pred_t)
        _,pred_label = torch.max(pred_sm,1) 
        plotImages(img*255, label*255., (pred_label.cpu().data.numpy().squeeze())*255.)
        
        lb = label_t.clone().long()
        pl = pred_label.clone()
        true_sum  = (lb == 1).sum().item()
        false_sum = (lb == 0).sum().item()
        true_acc  = ((lb * 100 - pl * -50) >= 150).sum().item() # only for class 1 
        false_acc = ((lb * 100 - pl * -50) == 0).sum().item() # only for class 0
        print(' Accuracy for class 1 : %.3f %%'% ((100 * true_acc / true_sum) if true_sum != 0 else 0))
        print(' Accuracy for class 0 : %.3f %%'% ((100 * false_acc / false_sum) if false_sum != 0 else 0))
        print(' Accuracy for total   : %.3f %%'% (\
            (100 * (false_acc + true_acc) / (false_sum + true_sum)) if (false_sum + true_sum) != 0 else 0)) 
        
        # Save Checkpoint for the lowest true acc
        acc = (100 * true_acc / true_sum)
        if (acc > max_acc):
            torch.save(net.state_dict(), join('/home/suhongk/sfuhome/', 'data') + '/CP_min.pth')
            max_acc = acc
            print('Checkpoint %d saved ! with max_acc : %.3f %%' % (epoch + 1, acc))
        
    # display total runtime     
    run_time = time.time() - start_time
    print(" Run time: %s min %s sec "% (run_time // 60, run_time %60))
    print(" Running is over !!!! \n\n")
    
    # displays test images with original and predicted masks after training
    loader.setMode('test')
    net.eval()
    true_sum = 0; false_sum = 0; true_acc = 0; false_acc = 0
    with torch.no_grad():
        for _, (img, label) in enumerate(loader):
            shape = img.shape
            label = label - 1
            img_torch = torch.from_numpy(img.reshape(1,1,shape[0],shape[1])).float()
            lbl_torch = torch.from_numpy(label.reshape(1, label.shape[0], label.shape[1])).long()
            if gpu:
                img_torch = img_torch.cuda()
                lbl_torch = lbl_torch.cuda()
            pred = net(img_torch)
            pred_sm = softmax(pred)
            _,pred_label = torch.max(pred_sm,1)   
            #plot
            plotImages(img*255, label*255., (pred_label.cpu().detach().numpy().squeeze())*255.)
            # Accuracy
            true_sum += (lbl_torch == 1).sum().item()
            false_sum += (lbl_torch == 0).sum().item()
            true_acc += ((lbl_torch * 100 -  pred_label * -50) >= 150).sum().item() # only for class 1 
            false_acc += ((lbl_torch * 100 -  pred_label * -50) == 0).sum().item() # only for class 0 
                
        print(' Accuracy for class 1 : %.6f %%'% ((100 * true_acc / true_sum) if true_sum != 0 else 0))
        print(' Accuracy for class 0 : %.6f %%'% ((100 * false_acc / false_sum) if false_sum != 0 else 0))
        print(' Accuracy for total   : %.6f %%'% (\
               (100 * (false_acc + true_acc) / (false_sum + true_sum)) if (false_sum + true_sum) != 0 else 0))

def plotImages(image, label, pred):
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.subplot(1, 3, 2)
    plt.imshow(label)
    plt.subplot(1, 3, 3)
    plt.imshow(pred)
    plt.show()
    
def getLoss(pred_label, target_label):
    p = softmax(pred_label)
    return cross_entropy(p, target_label)

def softmax(input):
    # todo: implement softmax function
    p = torch.exp(input) # N C H W
    p_sum = torch.sum(p, dim=1).view(p.shape[0], 1, p.shape[2], p.shape[3]) # N H W
    p = torch.div(p, (p_sum + 1e-9))
    return p

def cross_entropy(input, targets):
    # todo: implement cross entropy
    # Hint: use the choose function
    pl = choose(input, targets)
    ce = torch.mean(-1.0 * torch.log(pl + 1e-9))
    return ce

# Workaround to use numpy.choose() with PyTorch
def choose(pred_label, true_labels):
    size = pred_label.size()
    ind = np.empty([size[2]*size[3],3], dtype=int)
    i = 0
    for x in range(size[2]):
        for y in range(size[3]):
            ind[i,:] = [true_labels[x,y], x, y]
            i += 1

    pred = pred_label[0,ind[:,0],ind[:,1],ind[:,2]].view(size[2],size[3])

    return pred

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=5, type='int', help='number of epochs')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=2, type='int', help='number of classes')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='data/cells/', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=True, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_classes=args.n_classes)

    if args.load:
        print(args.load)
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net.cuda()
        cudnn.benchmark = True

    train_net(net=net,
        epochs=args.epochs,
        n_classes=args.n_classes,
        gpu=args.gpu,
        data_dir=args.data_dir)

