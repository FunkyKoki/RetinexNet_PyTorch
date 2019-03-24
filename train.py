import os
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import argparse
from model import DecomNet, RelightNet
from dataset import TheDataset
from loss import DecomLoss, RelightLoss
import tqdm


parser = argparse.ArgumentParser(description='RetinexNet args setting')

parser.add_argument('--use_gpu', dest='use_gpu', type=int, default=1, help='gpu flag, 1 for GPU and 0 for CPU')
parser.add_argument('--gpu_idx', dest='gpu_idx', default='0', help='GPU idx')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')

parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='number of total epoch')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of samples in one batch')
parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size')
parser.add_argument('--workers', dest='workers', type=int, default=8, help='num workers of dataloader')
parser.add_argument('--start_lr', dest='start_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--save_interval', dest='save_interval', type=int, default=20, help='save model every # epoch')

parser.add_argument('--checkpoint_dir', dest='ckpt_dir', default='./checkpoint', help='directory for checkpoints')
parser.add_argument('--save_dir', dest='save_dir', default='./test_results', help='directory for testing outputs')
parser.add_argument('--test_dir', dest='test_dir', default='./data/test/low', help='directory for testing inputs')
parser.add_argument('--decom', dest='decom', default=0,
                    help='decom flag, 0 for enhanced results only and 1 for decomposition results')

args = parser.parse_args()

if not os.path.exists(args.ckpt_dir):
    os.makedirs(args.ckpt_dir)

decom_net = DecomNet()
relight_net = RelightNet()

if args.use_gpu:
    decom_net = decom_net.cuda()
    relight_net = relight_net.cuda()
    cudnn.benchmark = True
    cudnn.enabled = True

lr = args.start_lr * np.ones([args.epoch])
lr[20:] = lr[0] / 10.0

decom_optim = torch.optim.Adam(decom_net.parameters(), lr=args.start_lr)
relight_optim = torch.optim.Adam(relight_net.parameters(), lr=args.start_lr)

train_set = TheDataset()

decom_criterion = DecomLoss()
relight_criterion = RelightLoss()


def train():
    decom_net.train()
    relight_net.train()

    for epoch in range(args.epoch):
        times_per_epoch, sum_loss = 0, 0.

        dataloader = torch.utils.data.Dataloader(train_set, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.workers, pin_memory=True)
        decom_optim.param_groups[0]['lr'] = lr[epoch]

        for data in tqdm.tqdm(dataloader):
            times_per_epoch += 1
            low_im, high_im = data
            low_im, high_im = low_im.cuda(), high_im.cuda()

            decom_optim.zero_grad()
            _, r_low, l_low = decom_net(low_im)
            _, r_high, l_high = decom_net(high_im)
            loss = decom_criterion(r_low, l_low, r_high, l_high, low_im, high_im)
            loss.backward()
            decom_optim.step()

            sum_loss += loss

        print('epoch: ' + str(epoch) + ' | loss: ' + str(sum_loss / times_per_epoch))
        if (epoch+1) % args.save_interval == 0:
            torch.save(decom_net.state_dict(), args.ckpt_dir + '/decom_' + str(epoch) + '.pth')

    torch.save(decom_net.state_dict(), args.ckpt_dir + '/decom_final.pth')

    for epoch in range(args.epoch):
        times_per_epoch, sum_loss = 0, 0.

        dataloader = torch.utils.data.Dataloader(train_set, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=args.workers, pin_memory=True)
        relight_optim.param_groups[0]['lr'] = lr[epoch]

        for data in tqdm.tqdm(dataloader):
            times_per_epoch += 1
            low_im, high_im = data
            low_im, high_im = low_im.cuda(), high_im.cuda()

            relight_optim.zero_grad()
            lr_low, r_low, _ = decom_net(low_im)
            l_delta = relight_net(lr_low.detach())
            loss = relight_criterion(l_delta, r_low.detach(), high_im)
            loss.backward()
            relight_optim.step()

            sum_loss += loss

        print('epoch: ' + str(epoch) + ' | loss: ' + str(sum_loss / times_per_epoch))
        if (epoch+1) % args.save_interval == 0:
            torch.save(relight_net.state_dict(), args.ckpt_dir + '/relight_' + str(epoch) + '.pth')

    torch.save(relight_net.state_dict(), args.ckpt_dir + '/relight_final.pth')


if __name__ == '__main__':
    train()
