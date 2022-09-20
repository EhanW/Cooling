import torch
from torch import nn
import argparse
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader, Dataset
from data import IndexedCIFAR10Train, IndexedCIFAR10Test
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
    parser.add_argument('--num_workers', default=4, type=int)
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
    parser.add_argument('--num-groups', default=10, type=int)
    parser.add_argument('--num-permutations', default=10, type=int)
    parser.add_argument('--retrain-epochs', default=5, type=int)
    return parser.parse_args()


class DataGroupShapley(object):
    def __init__(self, model, load_path, num_groups, num_permutations, retrain_epochs, train_set, test_set):
        self.model = model
        self.load_path = load_path
        self.num_groups = num_groups
        self.retrain_epochs = retrain_epochs
        self.train_set = train_set
        self.test_set = test_set
        self.num_permutations = num_permutations

        self.marginals_history = torch.zeros((0, num_groups))
        self.adv_marginals_history = torch.zeros((0, num_groups))

        self.train_loader = DataLoader(self.train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                       shuffle=True, pin_memory=True)
        self.test_loader = DataLoader(self.test_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                      shuffle=False, pin_memory=True)
        self.optimizer = SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

        self.init_model()
        self.init_acc, self.init_adv_acc = self.test()
        self.group_indices = self.split_groups()

    def run(self):
        for p in range(self.num_permutations):
            print(p)
            marginals, adv_marginals = self.one_permutation()
            self.marginals_history = torch.cat((self.marginals_history, marginals), dim=0)
            self.adv_marginals_history = torch.cat((self.adv_marginals_history, adv_marginals), dim=0)
        shapley_values = torch.mean(self.marginals_history, dim=0)
        adv_shapley_values = torch.mean(self.adv_marginals_history, dim=0)
        return shapley_values, adv_shapley_values

    def one_permutation(self):
        permutation = torch.randperm(self.num_groups)
        marginals = torch.zeros(self.num_groups)
        adv_marginals = torch.zeros(self.num_groups)

        new_score = self.init_acc
        new_adv_score = self.init_adv_acc
        for idx, index in enumerate(permutation):
            old_score = new_score
            old_adv_score = new_adv_score

            indices = self.group_indices[permutation[:idx+1], :].view(-1)

            retrain_loader = self.get_retrain_loader(indices)
            self.retrain(retrain_loader)
            new_score, new_adv_score = self.test()

            marginals[index] = new_score - old_score
            adv_marginals[index] = new_adv_score - old_adv_score
        return marginals, adv_marginals

    def init_model(self):
        ckpt = torch.load(self.load_path, map_location=args.device)
        self.model.load_state_dict(ckpt)

    def get_retrain_loader(self, indices):
        sampler = SubsetRandomSampler(indices)
        loader = DataLoader(self.train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                            sampler=sampler, pin_memory=True)
        return loader

    def split_groups(self):
        self.model.eval()
        sequential_train_loader = DataLoader(self.train_set, batch_size=args.batch_size, num_workers=args.num_workers,
                                             shuffle=False, pin_memory=True)
        losses = list()
        for data, target, index in sequential_train_loader:
            data, target = data.to(args.device), target.to(args.device)
            with torch.no_grad():
                adv_data = pgd_inf_test(self.model, data, target, args.epsilon, args.alpha, args.steps,
                                        args.random_start,
                                        args.restarts)
                adv_preds = self.model(adv_data)
                loss = F.cross_entropy(adv_preds, target, reduction='none')
                losses.append(loss)
        losses = torch.cat(losses, dim=0)
        indices = losses.sort()[1].cpu()
        group_capacity = int(args.total/self.num_groups)
        group_indices = [
            indices[k*group_capacity:(k+1)*group_capacity]
            for k in range(self.num_groups)
        ]
        group_indices = torch.stack(group_indices, dim=0)
        return group_indices

    def retrain(self, loader):
        self.init_model()
        self.model.train()
        for epoch in range(self.retrain_epochs):
            self.retrain_epoch(loader)

    def retrain_epoch(self, loader):
        for id, (data, target, index) in enumerate(loader):
            data, target = data.to(args.device), target.to(args.device)
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
        for data, target, index in self.test_loader:
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
    path = './logs/shapley/resnet18/at/ckpts/105.pth'
    model = eval(args.model_name)(args.num_classes).to(args.device)
    dgs = DataGroupShapley(model, load_path=path, num_groups=args.num_groups,
                           num_permutations=args.num_permutations, retrain_epochs=args.retrain_epochs,
                           train_set=IndexedCIFAR10Train(), test_set=IndexedCIFAR10Test())
    shapley, adv_shapley = dgs.run()
    print(shapley)