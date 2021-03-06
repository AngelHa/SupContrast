from __future__ import print_function

import sys
import os
import argparse
import time
import math
import numpy as np
import pandas as pd

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from torchvision.datasets import ImageFolder

from tqdm import tqdm
# from main_ce import set_loader
import tensorboard_logger as tb_logger
from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model
from networks.resnet_big import SupConResNet, LinearClassifier

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')


    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'path'], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')


    # other setting
    parser.add_argument('--output', type=str, default='prediction.csv',
                        help='output csv file')

    parser.add_argument('--ckpt', type=str, default='',
                        help='path to pre-trained model')
    parser.add_argument('--get_confidence', action='store_true',
                        help='get confidence in the output file')
    parser.add_argument('--get_probabilities', action='store_true',
                        help='get probabilities in the output file')

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/'

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'path':
        opt.n_cls = 5
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt

def set_loader(opt):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.mean)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)


    data_transform = transforms.Compose([
        transforms.Resize(size=opt.size),
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=data_transform)
    elif opt.dataset == 'cifar100':
        dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=data_transform)
    elif opt.dataset == 'path':
        dataset = datasets.ImageFolder(root=opt.data_folder,
                                            transform=data_transform)
    else:
        raise ValueError(opt.dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    return data_loader

def set_model(opt):
    model = SupConResNet(name=opt.model)
    criterion = torch.nn.CrossEntropyLoss()
    
    classifier = LinearClassifier(name=opt.model, num_classes=opt.n_cls)

    ckpt = torch.load(opt.ckpt, map_location='cpu')
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
        classifier.load_state_dict(ckpt['classifier'].state_dict())
        
    return model, classifier

def predict(data_loader, model, classifier, opt):
    """validation"""
    model.eval()
    classifier.eval()

    all_output = [] 
    with torch.no_grad():
        end = time.time()
        for idx, (images, _) in tqdm(enumerate(data_loader)):
            images = images.float().cuda()

            # forward
            output = classifier(model.encoder(images))
            all_output.append(output.cpu())

    return torch.cat(all_output)


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    data_loader = set_loader(opt)

    # build model and classifier
    model, classifier = set_model(opt)
    
    # predict routine
    outputs = predict(data_loader, model, classifier, opt)
    
    outputs = torch.nn.functional.softmax(outputs, dim=1)
    values, labels = outputs.max(dim=1)
    
    classes = np.array(['chip', 'fines', 'lump', 'mixed', 'pellets'])
    # classe_to_idx = {'chip': 0, 'fines': 1, 'lump': 2, 'mixed': 3, 'pellets': 4}
    files  = [path.split('/')[-1] for path,_ in data_loader.dataset.samples]

    df = pd.DataFrame.from_dict({
        'files': files,
        'labels': classes[labels],
    })

    if opt.get_confidence:
        df['confidence'] = values.numpy()
    
    if opt.get_probabilities:
        df[classes] = outputs.numpy()
        
    
    df.to_csv(opt.output, index=None)


if __name__ == '__main__':
    main()
