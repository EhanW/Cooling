from networks import *
from data import get_train_loader, get_test_loader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import os
from utils import *
import numpy as np
import argparse


def get_args():
    model_names = [
        'resnet20', 'resnet32', 'resnet44', 'resnet56',
        'resnet34', 'resnet50',
        'vgg16', 'wide_resnet32_10'
        'preact_resnet18', 'preact_resnet34', 'preact_resnet50',
    ]
    parser = argparse.ArgumentParser('CIFAR10-PGD-AT-COOLING')
    parser.add_argument('--adv-train', default=True)
    parser.add_argument('--model-name', default='resnet18', choices=model_names)
    parser.add_argument('--batch-size', default=128, type=float)
    parser.add_argument('--num_classes', default=10, type=int)
    parser.add_argument('--epsilon', default=8/255, type=float)
    parser.add_argument('--alpha', default=2/255, type=float)
    parser.add_argument('--steps', default=10, type=int)
    parser.add_argument('--random-start', default=True)
    parser.add_argument('--restarts', default=1, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight-decay', default=5e-4, type=float)
    parser.add_argument('--milestones', default=(100, 150), type=tuple[int])
    parser.add_argument('--gamma', default=0.1, type=float)
    parser.add_argument('--total', default=50000, type=int, help='the number of samples in the training set')

    parser.add_argument('--device', default='cuda:2', type=str)
    parser.add_argument('--cooling-ratio', default=0, type=float, help='the max proportion of training samples to be cooled')
    parser.add_argument('--cooling-interval', default=300, type=int, help='the number of epochs for which the cooling procedure lasts')
    parser.add_argument('--cooling-start-epoch', default=300, type=int, help='the start epoch of cooling')

    return parser.parse_args()


def sort_losses(indices, losses):
    indices = torch.cat(indices, dim=0)
    losses = torch.cat(losses, dim=0)
    # ?????????????????????losses
    indices = indices.sort()[1]
    losses = losses[indices]
    return losses


def get_cooling_indices(losses):
    # ????????????????????????
    indices = torch.nonzero(states).view(-1)
    total_losses = losses[indices]
    # ??????losses?????????cooling_new??????????????????
    losses_least = total_losses.sort()[1][:cooling_new]
    cooling_indices = indices[losses_least]
    return cooling_indices


def get_avg_loss(losses, indices):
    total_losses = losses[indices]
    avg_loss = total_losses.mean().item()
    return avg_loss


def update_states(cooling_indices):
    if len(cooling_list) == args.cooling_interval:
        retrieve_indices = cooling_list.pop(0)
        states[retrieve_indices] = True
    cooling_list.append(cooling_indices)
    states[cooling_indices] = False


def pgd_at_cooling():
    save_path = os.path.join(path, 'ckpts')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    best_state = np.array([-1, 0.7, 0.3])  # (-best_loss, best_acc, best_adv_acc)

    for epoch in range(args.epochs):
        model.train()
        indices, losses = list(), list()
        for id, (data, target, index) in enumerate(train_loader):
            data, target = data.to(args.device), target.to(args.device)
            if args.adv_train:
                data = pgd_inf(model, data, target, args.epsilon, args.alpha, args.steps, args.random_start)
            loss = F.cross_entropy(model(data), target, reduction='none')

            indices.append(index)
            losses.append(loss.detach().cpu())

            loss *= states[index].to(args.device)
            num = states[index].sum().item()
            if num > 0:
                loss = loss.sum()/num
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # ????????????????????????losses
        losses = sort_losses(indices, losses)
        # ?????????????????????????????????loss
        avg_loss = get_avg_loss(losses, torch.nonzero(states))

        # ????????????
        if epoch + 1 >= args.cooling_start_epoch:
            # ????????????????????????????????????
            cooling_indices = get_cooling_indices(losses)
            # ????????????????????????????????????loss
            cooling_avg_loss = get_avg_loss(losses, cooling_indices)
            # ??????loss??????rewarming_list
            rewarming_list[epoch+args.cooling_interval] = cooling_avg_loss
            logger.add_scalar('cooling_loss', cooling_avg_loss, global_step=epoch)
            # ?????????????????????
            if len(cooling_list) == args.cooling_interval:
                # ???????????????????????????????????????????????????loss
                rewarming_avg_loss = get_avg_loss(losses, cooling_list[0])
                # ??????????????????????????????loss??????,?????????
                rewarming = rewarming_avg_loss - rewarming_list[epoch]
                rewarming_list[epoch] = rewarming
                logger.add_scalar('rewarming', rewarming, global_step=epoch)
            # ??????cooling_list, states
            update_states(cooling_indices)
        # ???????????????
        test_acc, test_adv_acc = pgd_test(test_loader)
        logger.add_scalar('avg loss', avg_loss, global_step=epoch)
        logger.add_scalar('acc', test_acc, global_step=epoch)
        logger.add_scalar('adv acc', test_adv_acc, global_step=epoch)

        if epoch >= args.milestones[0]:
            temp_state = np.array([-avg_loss, test_acc, test_adv_acc])
            cur_state = np.max((temp_state, best_state), axis=0)
            if (cur_state != best_state).any():
                torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}.pth'))
                best_state = cur_state
        if epoch in [100, 105, 110, 150, 155, 160]:
            torch.save(model.state_dict(), os.path.join(save_path, f'{epoch}.pth'))

    torch.save(model.state_dict(), os.path.join(save_path, 'end.pth'))


def pgd_test(loader):
    model.eval()
    total = 0
    correct = 0
    adv_correct = 0
    for data, target, index in loader:
        data, target = data.to(args.device), target.to(args.device)
        total += len(data)
        with torch.no_grad():
            correct += model(data).argmax(1).eq(target).sum().item()
            # adv_data = pgd_inf(model, data, target, args.epsilon, args.alpha, args.steps, args.random_start)
            adv_data = pgd_inf_test(model, data, target, args.epsilon, args.alpha, args.steps, args.random_start,
                                    args.restarts)
            adv_correct += model(adv_data).argmax(1).eq(target).sum().item()
    acc = correct/total
    adv_acc = adv_correct/total
    return acc, adv_acc


if __name__ == '__main__':
    args = get_args()

    # ??????????????????????????????????????????????????????
    cooling_new = int(args.total*args.cooling_ratio/args.cooling_interval)
    # ??????????????????, true:????????????, false:?????????
    states = torch.ones(args.total, dtype=torch.bool)
    # ????????????, ???????????????????????????????????????????????????
    cooling_list = list()
    # ????????????, ?????????????????????????????????????????????loss??????????????????
    rewarming_list = [0 for _ in range(args.epochs+args.cooling_interval)]

    test_loader = get_test_loader(args.batch_size)
    train_loader = get_train_loader(args.batch_size)

    model = eval(args.model_name)(num_classes=args.num_classes).to(args.device)
    optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    scheduler = MultiStepLR(optimizer=optimizer, milestones=args.milestones, gamma=args.gamma)

    path = os.path.join('logs/'+args.model_name, f'{args.cooling_ratio}_{args.cooling_interval}_{args.cooling_start_epoch}')
    os.makedirs(path, exist_ok=True)
    logger = SummaryWriter(log_dir=path)
    logger.add_text('args', str(args))
    pgd_at_cooling()
    logger.close()
