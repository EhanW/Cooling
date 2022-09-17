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
    parser.add_argument('--model-name', default='preact_resnet18', choices=model_names)
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

    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--cooling-ratio', default=20/100, type=float, help='the max proportion of training samples to be cooled')
    parser.add_argument('--cooling-interval', default=10, type=int, help='the number of epochs for which the cooling procedure lasts')
    parser.add_argument('--cooling-start-epoch', default=20, type=int, help='the start epoch of cooling')

    return parser.parse_args()


def sort_losses(indices, losses):
    indices = torch.cat(indices, dim=0)
    losses = torch.cat(losses, dim=0)
    # 按索引重新排序losses
    indices = indices.sort()[1]
    losses = losses[indices]
    return losses


def get_cooling_indices(losses):
    # 去除冷却中的数据
    indices = torch.nonzero(states)
    total_losses = losses[indices]
    # 获得losses最小的cooling_new个数据的索引
    losses_least = total_losses.sort()[1][:cooling_new]
    cooling_indices = indices[losses_least]
    return cooling_indices


def get_avg_loss(losses, indices):
    # indices = torch.nonzero(states)
    total_losses = losses[indices]
    avg_loss = total_losses.mean().item()
    return avg_loss


def update_states(cooling_indices):
    if len(cooling_list) == args.cooling_interval:
        retrieve_indices = cooling_list.pop(0)
        states[retrieve_indices] = ~states[retrieve_indices]
    cooling_list.append(cooling_indices)
    states[cooling_indices] = ~states[cooling_indices]


def pgd_at_cooling():
    save_path = os.path.join(path, 'ckpts')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    best_state = np.array([-1, 0.7, 0.3])  # (-best_loss, best_acc, best_adv_acc)

    for epoch in range(args.epochs):
        model.train()
        indices, losses = list(), list()

        for data, target, index in train_loader:
            data, target = data.to(args.device), target.to(args.device)
            if args.adv_train:
                data = pgd_inf(model, data, target, args.epsilon, args.alpha, args.steps, args.random_start)
            loss = F.cross_entropy(model(data), target, reduction='none')

            indices.append(index)
            losses.append(loss.detach().cpu())

            # 计算mean loss时去除冷却中的数据
            loss = loss[torch.nonzero(states[index]).to(args.device)].mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        # 按照索引重新排序losses
        losses = sort_losses(indices, losses)
        # 计算本轮中使用到的平均loss
        avg_loss = get_avg_loss(losses, torch.nonzero(states))

        # 开始冷却
        if epoch + 1 >= args.cooling_start_epoch:
            # 计算需要被冷却的数据索引
            cooling_indices = get_cooling_indices(losses)
            # 计算将被冷却的数据的平均loss
            cooling_avg_loss = get_avg_loss(losses, cooling_indices)
            # 将该loss写入rewarming_list
            rewarming_list[epoch+args.cooling_interval] = cooling_avg_loss

            # 冷却过程完成后
            if len(cooling_list) == 5:
                # 计算即将被释放的冷却数据目前的平均loss
                rewarming_avg_loss = get_avg_loss(losses, cooling_list[0])
                # 计算当前与被冷却时的loss之差,并记录
                rewarming = rewarming_avg_loss - rewarming_list[epoch]
                rewarming_list[epoch] = rewarming
                logger.add_scalar('rewarming', rewarming, global_step=epoch)
            # 更新cooling_list, states
            update_states(cooling_indices)

        # 测试与记录
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

    # 计算每个轮次中需要被冷却的数据的个数
    cooling_new = int(args.total*args.cooling_ratio/args.cooling_interval)
    # 标记数据状态, true:未被冷却, false:冷却中
    states = torch.ones(args.total, dtype=torch.bool)
    # 冷却列表, 其每个元素是该轮次中冷却数据的索引
    cooling_list = list()
    # 回温列表, 其每个元素是该轮次数据冷却前后loss增长的平均值
    rewarming_list = [0 for _ in range(args.epochs)]

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
