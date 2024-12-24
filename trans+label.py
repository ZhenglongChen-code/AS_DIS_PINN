import os
import random
import sys
import time
import copy
import torch
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.Mesh import MeshGrid
from src.Net_structure import *
from src.plot import plot_10_figs, show_filed
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler


class SequenceModel(nn.Module):
    def __init__(self, model_list):
        super(SequenceModel, self).__init__()
        self.time_step = len(model_list)
        layer_list = []
        for t in range(self.time_step):
            layer_list.append(('step_%d' % t, model_list[t]))

        self.layer = nn.Sequential(OrderedDict(layer_list))
        self.model_list = model_list

    def predict(self, time_step, input_fig):
        press_out = self.model_list[time_step-1](input_fig)
        return press_out


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.data = X
        # self.label = y

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


class PressHistory:
    def __init__(self):
        self.data = []
        self.batch_num = 1

    def add(self, batch_tensor):
        """
        actually we only need store 1 batch press data as an example,
        to be simple, just add last batch press tensor.
        :param batch_tensor: [batch1_press, batch2_press, ...]
        :return: self.data = [batch1_history, batch2_history, ...]
        """
        if len(self.data) == 0:
            self.batch_num = len(batch_tensor)
            self.data = [batch_tensor[i].unsqueeze(0) for i in range(self.batch_num)]

        else:
            for i in range(self.batch_num):
                self.data[i] = torch.concat((self.data[i], batch_tensor[i].unsqueeze(0)), dim=0)


def read_file(filepath):
    train_data = np.load(filepath)
    train_permeability, test_permeability = train_data['train_permeability'], train_data['test_permeability']
    train_label_press, test_label_press = train_data['train_press'], train_data['test_press']
    return train_permeability, test_permeability, train_label_press, test_label_press


def seed_everything(seed=227):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--id', default=0, type=int,
                    help='select a gpu to train NN')
parser.add_argument('-g', '--logdir', default='./logs/trans400+label100_4well')
parser.add_argument('--fig_dir', default='./trans400+label100_4well/', type=str)
parser.add_argument('-t', '--dt', default=30000, type=int,
                    help='time interval of PDE model')
parser.add_argument('--max_iter', default=500, type=int,
                    help='max iters for training process of NN')
parser.add_argument('--train_data', default='./dataset/sg_4well.npz', type=str)
parser.add_argument('-n', '--nt', default=150, type=int,
                    help='how long will the NN model predict')
parser.add_argument('--bs', default=400, type=int, help='batch size')
parser.add_argument('--lr', default=3e-3, type=float, help='learning rate')
parser.add_argument('--interval', default=100, type=int)
parser.add_argument('-mn', '--model_name', default='trans400+label100_4well4', type=str)
# 3: 使用全局最大最小值训练； 4: 使用p_scale 训练；
parser.add_argument('-ps', '--pinn_size', default=400, type=int)
parser.add_argument('-ls', '--label_size', default=100, type=int)
parser.add_argument('-wn', '--well_num', default=4, type=int)
parser.add_argument('-pt', '--pretrain', default=None,
                    type=str, help='weather use pretrain model')
parser.add_argument('--lam', default=1, type=float)

args = parser.parse_args()

if __name__ == '__main__':
    devices = check_devices()
    selected_gpu = devices[args.id]
    seed_everything(227)
    # init params and other condition
    nt = args.nt
    dt = args.dt
    max_iter = args.max_iter
    pinn_size = args.pinn_size
    label_size = args.label_size
    p_init, p_bc = 30 * 1e+6, None
    mu_o = 2e-3
    ct = 5e-8
    p_init = 30.0 * 1e6
    p_scale = 1e6  # 观察原始数据里面的不同样本之间差距大约是10^5数量级，希望网络输出的数据
    qw = 0.0005
    chuk = 5e-15
    poro = 0.1
    rw = 0.05
    SS = 3
    length = 3000
    cs = ct * 3.14 * rw * rw * length
    bhp_constant = 20 * 1e6
    nx, ny, nz = 20, 20, 1  # x, y, z 方向的cell数目, 这个方向顶点个数要比cell数目多一个
    if args.well_num == 1:
        well_mark = [0]  # 源的位置
    else:
        well_mark = [0, 19, 380, 399]

    # if args.bs < args.pinn_size:
    #     batch_size = args.bs
    # else:
    #     batch_size = args.pinn_size

    batch_size = min(args.bs, args.pinn_size)
    # 1. load data
    file = args.train_data
    train_permeability, test_permeability, train_press, test_press = read_file(file)

    label_time = train_press.shape[1]  #
    print('label press has {} time step, model need to train {} time step'.format(label_time, nt))
    assert len(train_press) >= args.nt, 'label press length < nt'

    # 1.1 save data into logs
    writer = SummaryWriter(comment='discriminate_data', log_dir=args.logdir)

    # 1.2 compute trans_matrix as training data, train_fig: [n, 4, 20, 20]
    mesh = MeshGrid(nx, ny, nz, train_permeability[0].flatten(), mu_o, ct,
                    poro, p_init, p_bc, bhp_constant, devices=[selected_gpu])

    # define a function to compute trans_matrix
    def compute_trans(perme, desc='generate mesh and trans_matrix'):
        trans_matrix = []
        for perm in tqdm(perme, desc=desc):
            mesh = MeshGrid(nx, ny, nz, perm.flatten(), mu_o, ct,
                            poro, p_init, p_bc, bhp_constant, devices=[selected_gpu])
            if len(trans_matrix) == 0:
                trans_matrix = mesh.trans_matrix.detach().flatten()
                permeability = torch.tensor(perm)
            else:
                trans_matrix = torch.vstack((trans_matrix, mesh.trans_matrix.detach().flatten()))
                permeability = torch.vstack((permeability, torch.tensor(perm)))

        return trans_matrix

    # select pinn size
    train_trans_matrix = compute_trans(train_permeability, desc='compute train trans_matrix')
    train_trans_matrix = train_trans_matrix.reshape(-1, 4, ny, nx)

    scaled_params = 1e-12  # scaled tans_matrix into [0, 1000]

    # select label data,
    train_label_fig = train_trans_matrix[:label_size].clone().to(selected_gpu) / scaled_params
    train_label_press = torch.tensor(train_press[:label_size], dtype=torch.float64).to(selected_gpu)

    lam = args.lam
    print('lam:{:4f}'.format(lam))
    train_loader = DataLoader(MyDataset(train_trans_matrix[label_size:label_size+pinn_size] / scaled_params),
                              batch_size=batch_size, shuffle=False)  # pinn data don't contain label data
    min_label_pt = torch.min(train_label_press, dim=0).values
    max_label_pt = torch.max(train_label_press, dim=0).values
    min_pt = min_label_pt.min(dim=0).values
    max_pt = max_label_pt.max(dim=0).values

    # print('pinn data size:{}')
    neighbor_idx = mesh.neighbor_vectors  # constant value
    test_trans_matrix = compute_trans(test_permeability, desc='compute test trans_matrix')
    test_fig = test_trans_matrix.reshape(-1, 4, nx, ny).to(selected_gpu) / scaled_params
    # NN params
    criterion = nn.MSELoss(reduction='none')
    L1_loss = nn.L1Loss(reduction='none')
    b1 = nn.Sequential(nn.Conv2d(4, 20, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))  # 1x20x20 --> 20x10x10
    b2 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
    # b3 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
    b4 = nn.Sequential(*resnet_block(20, 40, 2, first_block=False))
    b5 = nn.Sequential(*resnet_block(40, 80, 2, first_block=False))

    input_size = nx * ny * nz  # the size of input tensor,
    output_size = input_size
    if args.pretrain is not None:
        pretrain_file = args.pretrain
        model = torch.load(pretrain_file, map_location=lambda storage, loc: storage)
        model = model.to(selected_gpu)
    else:
        model = nn.Sequential(b1, GAM_Attention(20, 20),
                              b2, b4,
                              b5, GAM_Attention(80, 80), nn.Flatten(),
                              # nn.Dropout(p=0.15),
                              nn.Linear(720, output_size)).to(torch.float64).to(selected_gpu)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-5)
    # scheduler = StepLR(optimizer, step_size=500, gamma=0.95)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=1e-3)
    # train model
    nt = args.nt
    interval = args.interval
    model_list = []
    press_history = PressHistory()
    gs = 1
    for t in range(nt):
        batch_press = []
        batch = 0
        best_model = None
        min_loss = 1e10  # in each time step, reset min_loss
        min_iter_time = None  # record the ith iter of best model
        for input_fig in train_loader:
            input_fig = input_fig.to(selected_gpu)
            trans = input_fig.view(-1, 4*nz*ny*nx)

            if t == 0:
                p_last = torch.tile(mesh.press, (len(input_fig), 1)).to(selected_gpu)
            else:
                p_last = press_history.data[batch][-1]
                model = model_list[-1]

            max_iter = 2 * args.max_iter if t < 1 else args.max_iter
            # max_iter = args.max_iter
            for i in tqdm(range(max_iter), desc='training'):
                optimizer.zero_grad()
                model.train()
                # calculate PINN loss
                # p_next = model(input_fig) * p_scale
                p_next = model(input_fig) * (max_pt - min_pt) + min_pt

                res = torch.zeros_like(p_next)
                res[:] = res[:] + (mesh.cell_volume * mesh.porosity * mesh.ct / dt) * (p_next[:] - p_last[:])
                # for j in range(batch_size):  # 把trans_matrix 改成4个方向的tensor可以加速。
                for d in range(4):
                    res[:] = (res[:] - scaled_params * trans[:, d * 400:(d + 1) * 400] *
                              (p_next[:, neighbor_idx[d]] - p_next[:]))

                res[:, well_mark] += mesh.PI * (p_next[:, well_mark] - mesh.pwf)
                # res[:, 0] = res[:, 0] + mesh.PI * (p_next[:, 0] * p_init - mesh.pwf)
                loss1 = criterion(res, torch.zeros_like(res)).mean(dim=0).sum()

                # compute label loss
                # p_pre = model(train_label_fig) * p_scale
                p_pre = model(train_label_fig)  # * (max_pt - min_pt) + min_pt

                loss2 = criterion(p_pre, (train_label_press[:, t, :] - min_pt) / (max_pt - min_pt)).mean(dim=0).sum()

                loss = loss1 + lam * loss2
                loss.backward()  # 仅在必要时使用 retain_graph=True
                optimizer.step()
                scheduler.step()
                if loss < min_loss:
                    min_loss = loss.item()
                    min_iter_time = i+1
                    best_model = copy.deepcopy(model)
                    # best_p_next = p_next.clone()

                if (i + 1) % interval == 0:
                    print('\ntime step:{}, batch:{}, train step:{:d}, '
                          'PINN loss:{}, label loss:{}'.format(t, batch, i+1, loss1.item(), loss2.item()))
                    print('now, the minimum loss at {}th iter, and min_loss is{}'.format(min_iter_time, min_loss))

                    # model.eval()
                    # test_p = model(test_fig).detach()
                    # 记录下一个时间步的压力标签，
                    label_t_press = torch.tensor(test_press[:, t, :]).to(selected_gpu)
                    writer.add_scalar('loss_sum', loss, global_step=gs)
                    writer.add_scalars('PINN & label loss', {'pinn loss': loss1, 'label loss': loss2}, global_step=gs)

                    # best model L1 loss
                    # test_p_b = best_model(test_fig).detach() * p_scale
                    test_p_b = best_model(test_fig).detach() * (max_pt - min_pt) + min_pt
                    test_loss_b = L1_loss(test_p_b, label_t_press) / label_t_press.abs()
                    re_L1_b = test_loss_b.mean()
                    print('best model, test L1 relative error: {}'.format(re_L1_b.item()))
                    writer.add_scalar('best model relative L1 mean error', re_L1_b, global_step=gs)

                    # best model L1 loss in well condition
                    re_L1_well = test_loss_b[:, well_mark].mean()
                    print('best model, test L1 relative error in well grid: {}'.format(re_L1_well.item()))
                    writer.add_scalar('best model relative L1 well error', re_L1_well, global_step=gs)
                    gs += 1

                if loss < 1e-17:
                    print('time step:{}, batch:{}, train step:{:d}, loss{}'.format(t, batch, i + 1, loss.item()))
                    break

            # p_next[:, 0] = p_bc / p_init
            best_p_next = best_model(input_fig) * (max_pt - min_pt) + min_pt
            batch_press.append(best_p_next.detach())
            batch += 1
            # show_filed(best_p_next[0].detach(), nx, ny,args.model_name + '{}_wellpress_t_{}'.format(args.well_num, t+1), show=False)
            rand_idx = [random.randint(0, label_size-1) for _ in range(10)]
            plot_10_figs(best_p_next[rand_idx, :], nx, ny,
                         args.fig_dir + 'train_' + args.model_name + 'sample_batch{}_t_{}.png'.format(batch, t+1), id_num=rand_idx)

        press_history.add(batch_press)
        model_list.append(best_model)
        # 防止断电等意外，每一步保存一次模型，
        sequence_model = SequenceModel(model_list)
        torch.save(sequence_model.model_list, './model/'+args.model_name+'.pth')
        print('already save {} time step model'.format(t+1))
        # plot_10_figs()
        if (t + 1) % 10 == 0:
            press_sample = press_history.data[-1][:, 0, :]  # [last_batch][time_step, sample_id, press_dim]
            plot_10_figs(press_sample[t-9: t+1], nx, ny,
                         args.fig_dir + 'train_' + args.model_name + '_t_{}.png'.format(t+1))

    # sequence_model = SequenceModel(model_list)
    # torch.save(sequence_model.model_list, './model/'+args.model_name+'.pth')

    # test model
    vali_history = PressHistory()
    vari_fig = test_fig[-2:-1]
    for t in range(nt):
        # p1 = sequence_model.predict(t+1, vari_fig) * p_scale
        p1 = sequence_model.predict(t+1, vari_fig) * (max_pt - min_pt) + min_pt
        vali_history.add(p1.detach())

        if (t + 1) % 10 == 0:
            vari_press = vali_history.data[-1]  # [time_step, 1, press_dim]
            plot_10_figs(vari_press[t - 9: t + 1], nx, ny,
                         args.fig_dir + 'vari_' + args.model_name + '_t_{}.png'.format(t + 1))
