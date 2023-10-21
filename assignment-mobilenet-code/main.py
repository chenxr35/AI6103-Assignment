import os
import torch
import numpy as np
from mobilenet import MobileNet
from utils import plot_loss_acc, plot_lr
from dataloader import get_train_valid_loader, get_test_loader
import logging

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(pred, y_a, y_b, lam):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main(args):

    os.makedirs("log", exist_ok=True)

    # logging
    log_file = f"log/lr{args.lr}_wd{args.wd}_eps{args.epochs}_scheduler{args.lr_scheduler}_mixup{args.mixup}.log"
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)

    # fix random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)

    # train val test
    # AI6103 students: You need to create the dataloaders youself
    train_loader, valid_loader = get_train_valid_loader(args.dataset_dir, args.batch_size, True, args.seed, save_images=args.save_images) 
    if args.test:
        test_loader = get_test_loader(args.dataset_dir, args.batch_size)

    # model
    model = MobileNet(100)
    print(model)
    model.cuda()

    # criterion
    criterion = torch.nn.CrossEntropyLoss().cuda()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=args.epochs)

    stat_training_loss = []
    stat_val_loss = []
    stat_training_acc = []
    stat_val_acc = []
    stat_lr = []
    for epoch in range(args.epochs):
        training_loss = 0
        training_acc = 0
        training_samples = 0
        val_loss = 0
        val_acc = 0
        val_samples = 0
        # training
        model.train()
        if args.mixup:
            for imgs, labels in train_loader:
                imgs = imgs.cuda()
                labels = labels.cuda()
                mixed_x, y_a, y_b, lam = mixup_data(imgs, labels)

                batch_size = mixed_x.shape[0]
                optimizer.zero_grad()
                logits = model.forward(mixed_x)
                loss = mixup_criterion(logits, y_a, y_b, lam)
                loss.backward()
                optimizer.step()
                _, top_class = logits.topk(1, dim=1)
                equals = top_class == labels.cuda().view(*top_class.shape)
                training_acc += torch.sum(equals.type(torch.FloatTensor)).item()
                training_loss += batch_size * loss.item()
                training_samples += batch_size
        else:
            for imgs, labels in train_loader:
                imgs = imgs.cuda()
                labels = labels.cuda()

                batch_size = imgs.shape[0]
                optimizer.zero_grad()
                logits = model.forward(imgs)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                _, top_class = logits.topk(1, dim=1)
                equals = top_class == labels.cuda().view(*top_class.shape)
                training_acc += torch.sum(equals.type(torch.FloatTensor)).item()
                training_loss += batch_size * loss.item()
                training_samples += batch_size
        # validation
        model.eval()
        for val_imgs, val_labels in valid_loader:
            batch_size = val_imgs.shape[0]
            val_logits = model.forward(val_imgs.cuda())
            loss = criterion(val_logits, val_labels.cuda())
            _, top_class = val_logits.topk(1, dim=1)
            equals = top_class == val_labels.cuda().view(*top_class.shape)
            val_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            val_loss += batch_size * loss.item()
            val_samples += batch_size
        assert val_samples == 10000
        # update stats
        stat_training_loss.append(training_loss/training_samples)
        stat_val_loss.append(val_loss/val_samples)
        stat_training_acc.append(training_acc/training_samples)
        stat_val_acc.append(val_acc/val_samples)
        stat_lr.append(scheduler.get_lr()[0])
        # print
        logging.info(f"Epoch {(epoch+1):d}/{args.epochs:d}.. Learning rate: {scheduler.get_lr()[0]:.4f}.. Train loss: {(training_loss/training_samples):.4f}.. Train acc: {(training_acc/training_samples):.4f}.. Val loss: {(val_loss/val_samples):.4f}.. Val acc: {(val_acc/val_samples):.4f}")
        # lr scheduler
        scheduler.step()
    # plot
    plot_loss_acc(stat_training_loss, stat_val_loss, stat_training_acc, stat_val_acc, args.fig_name)
    # test
    if args.test:
        test_loss = 0
        test_acc = 0
        test_samples = 0
        for test_imgs, test_labels in test_loader:
            batch_size = test_imgs.shape[0]
            test_logits = model.forward(test_imgs.cuda())
            test_loss = criterion(test_logits, test_labels.cuda())
            _, top_class = test_logits.topk(1, dim=1)
            equals = top_class == test_labels.cuda().view(*top_class.shape)
            test_acc += torch.sum(equals.type(torch.FloatTensor)).item()
            test_loss += batch_size * test_loss.item()
            test_samples += batch_size
        assert test_samples == 10000
        logging.info(f'Test loss: {test_loss/test_samples}')
        logging.info(f'Test acc: {test_acc/test_samples}')

    lr_fig_name = "learning_curve.png"
    plot_lr(stat_lr, lr_fig_name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='parameters')
    parser.add_argument('--dataset_dir',type=str, help='')
    parser.add_argument('--batch_size',type=int, help='')
    parser.add_argument('--epochs', type=int, help='')
    parser.add_argument('--lr',type=float, help='')
    parser.add_argument('--wd',type=float, help='')
    parser.add_argument('--fig_name',type=str, help='')
    parser.add_argument('--lr_scheduler', action='store_true')
    parser.set_defaults(lr_scheduler=False)
    parser.add_argument('--mixup', action='store_true')
    # parser.set_defaults(mixup=False)
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--save_images', action='store_true')
    parser.set_defaults(save_images=False)
    parser.add_argument('--seed', type=int, default=0, help='')
    args = parser.parse_args()
    print(args)
    main(args)
