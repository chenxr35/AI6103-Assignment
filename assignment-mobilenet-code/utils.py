import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def plot_loss_acc(train_loss, val_loss, train_acc, val_acc, fig_name):
    x = np.arange(len(train_loss))
    max_loss = max(max(train_loss), max(val_loss))

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.set_ylim([0,max_loss+1])
    lns1 = ax1.plot(x, train_loss, 'y-.', label='train_loss', linewidth=1.5)
    lns2 = ax1.plot(x, val_loss, 'g-.', label='val_loss', linewidth=1.5)
    # ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('accuracy')
    ax2.set_ylim([0,1.1])
    lns3 = ax2.plot(x, train_acc, 'b-.', label='train_acc', linewidth=1.5)
    lns4 = ax2.plot(x, val_acc, 'r-.', label='val_acc', linewidth=1.5)
    # ax2.tick_params(axis='y', labelcolor='tab:red')

    lns = lns1+lns2+lns3+lns4
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=0)

    # fig.tight_layout()
    plt.grid(True)

    os.makedirs('./diagram', exist_ok=True)
    plt.savefig(os.path.join('./diagram', fig_name))

    np.savez(os.path.join('./diagram', fig_name.replace('.png', '.npz')), train_loss=train_loss, val_loss=val_loss, train_acc=train_acc, val_acc=val_acc)

def plot_lr(lr, fig_name):
    plt.figure()
    plt.plot(range(len(lr)), lr)
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.title('Learning Rate Curve')
    plt.grid(True)
    plt.savefig(os.path.join('./diagram', fig_name))
    np.savez(os.path.join('./diagram', fig_name.replace('.png ', '.npz')), lr=lr)


