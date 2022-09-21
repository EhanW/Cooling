import os
import torch
from torch import nn
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms as T
from utils import *
import numpy as np
from torch.optim import SGD
from networks import *


def get_args():
    model_names = [
        'resnet18',
        'resnet20', 'resnet32', 'resnet44', 'resnet56',
        'resnet34', 'resnet50',
        'vgg16', 'wide_resnet32_10'
                 'preact_resnet18', 'preact_resnet34', 'preact_resnet50',
    ]
    parser = argparse.ArgumentParser('CIFAR10-PGD-AT-SCALE')
    parser.add_argument('--adv-train', default=True)
    parser.add_argument('--model-name', default='resnet18', choices=model_names)
    parser.add_argument('--batch-size', default=128, type=float)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--epsilon', default=8 / 255, type=float)
    parser.add_argument('--alpha', default=2 / 255, type=float)
    parser.add_argument('--steps', default=10, type=int)
    parser.add_argument('--random-start', default=True)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--total', default=50000, type=int, help='the number of samples in the training set')

    parser.add_argument('--device', default='cuda:2', type=str)
    parser.add_argument('--group-mode', default='probability', choices=['probability', 'quantity'])
    parser.add_argument('--num-groups', default=5, type=int)
    parser.add_argument('--num-permutations', default=15, type=int)
    parser.add_argument('--retrain-epochs', default=5, type=int)
    return parser.parse_args()


class DataGroupShapley(object):
    def __init__(self, model, load_path, num_groups, num_permutations, retrain_epochs, data_path):
        self.model = model

        self.load_path = load_path
        self.num_groups = num_groups
        self.retrain_epochs = retrain_epochs
        self.num_permutations = num_permutations
        self.data_path = data_path

        self.train_images, self.train_labels, self.train_loader, self.test_loader = self.prepare_data()
        print(self.train_images.shape)
        self.marginals_history = torch.zeros((0, num_groups))
        self.adv_marginals_history = torch.zeros((0, num_groups))
        self.optimizer = SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

        self.init_model()
        self.init_acc, self.init_adv_acc = self.test()
        self.group_indices = self.split_groups()

    def prepare_data(self):
        train_set = datasets.CIFAR10(root=self.data_path,
                                     download=False,
                                     train=True,
                                     transform=T.Compose([T.RandomCrop(32, 4),
                                                          T.RandomHorizontalFlip(),
                                                          T.ToTensor()]))
        test_set = datasets.CIFAR10(root=self.data_path,
                                    download=False,
                                    train=False,
                                    transform=T.ToTensor())

        train_loader = DataLoader(train_set, batch_size=args.total, shuffle=False, pin_memory=True)
        train_images, train_labels = next(iter(train_loader))
        train_images, train_labels = train_images.to(args.device), train_labels.to(args.device)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, pin_memory=True)
        return train_images, train_labels, train_loader, test_loader

    def run(self):
        for p in range(self.num_permutations):
            marginals, adv_marginals = self.one_permutation()
            self.marginals_history = torch.cat((self.marginals_history, marginals.view(1, -1)), dim=0)
            self.adv_marginals_history = torch.cat((self.adv_marginals_history, adv_marginals.view(1, -1)), dim=0)
        shapley_values = torch.mean(self.marginals_history, dim=0)
        adv_shapley_values = torch.mean(self.adv_marginals_history, dim=0)
        shapley_writer = open(os.path.join(save_path, 'values.txt'), mode='w')
        shapley_writer.write(
            'shapley values'+str(shapley_values)+'adv shapley values'+str(adv_shapley_values)
        )

    def one_permutation(self):
        permutation = torch.randperm(self.num_groups)
        marginals = torch.zeros(self.num_groups)
        adv_marginals = torch.zeros(self.num_groups)

        new_score = self.init_acc
        new_adv_score = self.init_adv_acc
        for idx, index in enumerate(permutation):
            old_score = new_score
            old_adv_score = new_adv_score

            indices = list()
            for i in permutation[:idx+1]:
                indices.append(self.group_indices[i])

            indices = torch.cat(indices, dim=0)
            self.retrain(indices)
            new_score, new_adv_score = self.test()

            marginals[index] = new_score - old_score
            adv_marginals[index] = new_adv_score - old_adv_score
        return marginals, adv_marginals

    def init_model(self):
        ckpt = torch.load(self.load_path, map_location=args.device)
        self.model.load_state_dict(ckpt)

    def split_groups(self):
        self.model.eval()

        losses = list()
        for data, target in self.train_loader:
            data, target = data.to(args.device), target.to(args.device)
            with torch.no_grad():
                adv_data = pgd_inf_test(self.model, data, target, args.epsilon, args.alpha, args.steps,
                                        args.random_start,
                                        args.restarts)
                adv_preds = self.model(adv_data)
                loss = F.cross_entropy(adv_preds, target, reduction='none')
                losses.append(loss)
        losses = torch.cat(losses, dim=0).cpu()

        group_indices = list()
        if args.group_mode == 'probability':
            probs = torch.exp(-losses)
            for i in range(args.num_groups):
                group_indices.append(
                    torch.nonzero((i/args.num_groups < probs) * (probs <= (i+1)/args.num_groups)).view(-1)
                )

        if args.group_mode == 'quantity':
            indices = losses.sort()[1]
            group_capacity = int(args.total/self.num_groups)
            for i in range(self.num_groups):
                group_indices.append(
                    indices[i*group_capacity:(i+1)*group_capacity]
                )
        return group_indices

    def retrain(self, indices):
        self.init_model()
        self.model.train()
        for epoch in range(self.retrain_epochs):
            self.retrain_epoch(indices)

    def retrain_epoch(self, indices):
        shuffle = torch.randperm(len(indices))
        images, labels = self.train_images[indices[shuffle]], self.train_labels[indices[shuffle]]
        num_batches = int(np.ceil(len(images)/args.batch_size))
        for batch_idx in range(num_batches):
            data = images[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]
            target = labels[batch_idx*args.batch_size:(batch_idx+1)*args.batch_size]

            if args.adv_train:
                data = pgd_inf(self.model, data, target, args.epsilon, args.alpha, args.steps, args.random_start)
            loss = F.cross_entropy(self.model(data), target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def test(self):
        self.model.eval()
        total = 0
        correct = 0
        adv_correct = 0
        for data, target in self.test_loader:
            data, target = data.to(args.device), target.to(args.device)
            total += len(data)
            with torch.no_grad():
                correct += self.model(data).argmax(1).eq(target).sum().item()
                adv_data = pgd_inf_test(self.model, data, target, args.epsilon, args.alpha, args.steps,
                                        args.random_start,
                                        args.restarts)
                adv_preds = self.model(adv_data)
                adv_correct += adv_preds.argmax(1).eq(target).sum().item()
        acc = correct / total
        adv_acc = adv_correct / total
        return acc, adv_acc


if __name__ == '__main__':
    args = get_args()
    load_path = './logs/shapley/resnet18/at/ckpts/105.pth'
    save_path = f'./logs/shapley/resnet18/{args.group_mode}/{args.num_groups}_{args.num_permutations}_{args.retrain_epochs}'
    os.makedirs(save_path, exist_ok=True)
    model = eval(args.model_name)(args.num_classes).to(args.device)
    dgs = DataGroupShapley(model, load_path=load_path, num_groups=args.num_groups,
                           num_permutations=args.num_permutations, retrain_epochs=args.retrain_epochs,
                           data_path='/data/yihan/datasets')
    shapley_writer = open(os.path.join(save_path, 'groups.txt'), mode='w')
    for group in dgs.group_indices:
        print(group.shape)
    dgs.run()
    