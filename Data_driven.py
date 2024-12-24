import os
import random
import sys
import time
import copy
from typing import Dict, Any

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
        press_out = self.model_list[time_step - 1](input_fig)
        return press_out


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


class PressHistory:
    """
    实际只需要记录上一时间步的最准确压力值，不需要记录所有时间步的压力，这样可以减少显存占用
    """

    def __init__(self, batch_num=1):
        self.data = [0 for _ in range(batch_num)]
        self.batch_num = batch_num

    def update(self, batch_tensor, batch_id):
        """
        第一次初始化时
        :param batch_tensor: [batch1_press, batch2_press, ...]
        :return: self.data = [batch1_history, batch2_history, ...]
        """
        self.data[batch_id] = batch_tensor
        # if self.batch_num == 1:
        #     press_next = batch_tensor
        #     if len(self.data) == 0:
        #         self.data = press_next.unsqueeze(0)
        #     else:
        #         self.data = torch.concat((self.data, press_next.unsqueeze(0)), dim=0)
        #
        # else:
        #
        #     if len(self.data) == 0:
        #         self.batch_num = len(batch_tensor)
        #         self.data = [batch_tensor[i].unsqueeze(0) for i in range(self.batch_num)]
        #
        #     else:
        #         for i in range(self.batch_num):
        #             self.data[i] = torch.concat((self.data[i], batch_tensor[i].unsqueeze(0)), dim=0)


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


def data_prepare(args):
    """
    prepare data for training
    :param args:
    :return:
    """
    train_permeability, test_permeability, train_press, test_press = read_file(args.train_data)
    train_press = torch.tensor(train_press, dtype=torch.float64)

    label_time = train_press.shape[1]  #
    print('label press has {} time step, model need to train {} time step'.format(label_time, nt))
    assert len(train_press) >= args.nt, 'label press length < nt'

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
    if args.trans_data is None:
        train_trans_matrix = compute_trans(train_permeability, desc='compute train trans_matrix')
        test_trans_matrix = compute_trans(test_permeability, desc='compute test trans_matrix')
    else:
        trans_data = torch.load(args.trans_data)
        train_trans_matrix = trans_data['train_trans_matrix']
        test_trans_matrix = trans_data['test_trans_matrix']

    train_trans_matrix = train_trans_matrix.reshape(-1, 4, ny, nx)
    global scaled_params
    scaled_params = 1e-12  # scaled tans_matrix into [0, 1000]

    # select train label data, using random index
    if args.random_index:
        print('using random index.')
        train_label_index = np.random.choice(range(len(train_permeability)), size=label_size, replace=False)
    else:
        print('using order data')
        train_label_index = list(range(0, label_size))

    train_label_fig = train_trans_matrix[train_label_index].clone().to(selected_gpu) / scaled_params
    train_label_press = train_press[train_label_index].to(selected_gpu)

    new_index = [i for i in range(len(train_permeability)) if i not in train_label_index]
    label_pair = {'label_fig': train_trans_matrix[new_index] / scaled_params, 'label_press': train_press[new_index]}

    # only using training data information
    min_label_pt = torch.min(train_label_press, dim=0).values
    max_label_pt = torch.max(train_label_press, dim=0).values
    global max_pt, min_pt
    min_pt = min_label_pt.min(dim=0).values.to(selected_gpu)
    max_pt = max_label_pt.max(dim=0).values.to(selected_gpu)

    test_fig = test_trans_matrix.reshape(-1, 4, nx, ny).to(selected_gpu) / scaled_params

    # constant value
    global cell_volume, porosity, neighbor_idx, PI, pwf
    cell_volume, porosity, neighbor_idx, PI, pwf = mesh.cell_volume, mesh.porosity, mesh.neighbor_vectors, mesh.PI, mesh.pwf

    params: dict[str, float | torch.Tensor] = {'label_pair': label_pair, 'max_pt': max_pt, 'min_pt': min_pt,
                                               'train_label_fig': train_label_fig,
                                               'train_label_press': train_label_press,
                                               'test_fig': test_fig, 'test_press': test_press,
                                               'scaled_params': scaled_params, 'neighbor_idx': neighbor_idx,
                                               'cell_volume': cell_volume, 'porosity': porosity}

    config = {'max_pt': max_pt, 'min_pt': min_pt, 'scaled_params': scaled_params, 'neighbor_idx': neighbor_idx}
    torch.save(config, './dataset/config_' + args.model_name + '.pth')
    # torch.save({'test_fig': test_fig, 'test_press': test_press}, './dataset/testdata.pth')
    print('data already prepared\n')
    return params


class AdaptiveSampler:
    def __init__(self, label_fig, label_press):
        """
        :param label_fig:
        :param label_press: current label press
        """
        self.label_fig = label_fig
        self.label_press = label_press

    def sample(self, net, sample_num, criterion):
        """
        select new samples by residual
        :param net:
        :param sample_num:
        :param criterion:
        :return:
        """
        X = self.label_fig.to(selected_gpu)
        y = self.label_press.to(selected_gpu)
        net.eval()
        p_pre = net(X)
        error = criterion(p_pre, y).sum(dim=1)
        values, indices = torch.topk(error, sample_num)
        new_index = [i for i in range(len(X)) if i not in indices]
        # update set
        self.label_fig = X[new_index]
        self.label_press = y[new_index]
        return X[indices], y[indices]


def compute_pinn_loss(net, optimizer, criterion, input_fig, trans, p_last):
    optimizer.zero_grad()
    net.train()
    # calculate PINN loss
    p_next = net(input_fig) * (max_pt - min_pt) + min_pt

    res = torch.zeros_like(p_next)
    res[:] = res[:] + (cell_volume * porosity * ct / dt) * (p_next[:] - p_last[:])
    # 把trans_matrix 改成4个方向的tensor可以加速。
    for d in range(4):
        res[:] = (res[:] - scaled_params * trans[:, d * 400:(d + 1) * 400] *
                  (p_next[:, neighbor_idx[d]] - p_next[:]))

    res[:, well_mark] += PI * (p_next[:, well_mark] - pwf)
    # res[:, 0] = res[:, 0] + mesh.PI * (p_next[:, 0] * p_init - mesh.pwf)
    loss = criterion(res, torch.zeros_like(res))  # .mean(dim=0).sum()
    loss.backward()  # 仅在必要时使用 retain_graph=True
    optimizer.step()
    return loss


def compute_label_loss(net, optimizer, criterion, input_fig, p_label):
    optimizer.zero_grad()
    net.train()
    # calculate PINN loss
    p_next = net(input_fig) * (max_pt - min_pt) + min_pt
    loss = criterion(p_next / p_init, p_label / p_init)
    loss.backward()  # 仅在必要时使用 retain_graph=True
    optimizer.step()
    return loss


def train(args, params):
    """
    :param args:
    :param params:
    :return:updated model,
    """
    train_label_fig, train_label_press = params['train_label_fig'], params['train_label_press']
    criterion = nn.MSELoss(reduction='mean')
    L1_loss = nn.L1Loss(reduction='none')
    # L1_mean = nn.MSELoss(reduction='mean')

    input_size = nx * ny * nz  # the size of input tensor,
    output_size = input_size
    model_name = args.model_name
    if args.pretrain is not None:
        pt_model_name = args.pretrain
        model = torch.load('./model/' + pt_model_name + '.pth', map_location=lambda storage, loc: storage)
    else:
        b1 = nn.Sequential(nn.Conv2d(4, 20, kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))  # 1x20x20 --> 20x10x10
        b2 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
        # b3 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
        b4 = nn.Sequential(*resnet_block(20, 40, 2, first_block=False))
        b5 = nn.Sequential(*resnet_block(40, 80, 2, first_block=False))
        solver = nn.Sequential(b1, GAM_Attention(20, 20),
                               b2, b4,
                               b5, GAM_Attention(80, 80), nn.Flatten(),
                               # nn.Dropout(p=0.15),
                               nn.Linear(720, output_size)).to(torch.float64).to(selected_gpu)

    press_his = PressHistory(batch_num=1)
    gs = 1
    max_iter = args.max_iter
    new_model_list = []
    sample_num_list = [100, 150, 50, 50]  # map(int, args.sample_list)  #
    print(f'sample num list:{sample_num_list}')
    for t in range(nt):
        if args.pretrain is not None:
            layer_t = model[t].to(selected_gpu)
        else:
            layer_t = solver
        # 只微调最后一层的参数
        optimizer_t = torch.optim.AdamW(layer_t.parameters(), lr=args.lr, weight_decay=1e-6)
        min_loss = 1e10
        input_fig, input_label = train_label_fig, train_label_press[:, t, :]

        # prepare X, y pair, y is label_press of current time step
        adaptive_sampler = AdaptiveSampler(params['label_pair']['label_fig'], params['label_pair']['label_press'][:, t, :])

        for i in tqdm(range(max_iter), desc='training'):
                
            loss = compute_label_loss(layer_t, optimizer_t, criterion, input_fig,
                                      input_label)
    
            if loss < min_loss:
                min_loss = loss.item()
                min_iter_time = i + 1
                best_model = copy.deepcopy(layer_t)
                # best_p_next = p_next.clone()

            if (i + 1) % interval == 0:

                print('\ntime step:{}, train step:{:d}, '
                        'label loss:{}'.format(t, i + 1, loss.item()))
                print('now, the minimum loss at {}th iter, and min_loss is{}'.format(min_iter_time, min_loss))

                # 记录下一个时间步的压力标签，
                label_t_press = torch.tensor(test_press[:, t, :]).to(selected_gpu)
                writer.add_scalars('PINN & label loss',
                                    {'pinn loss': 0, 'label loss': loss}, global_step=gs)

                # best model L1 loss
                test_p_b = best_model(test_fig).detach() * (max_pt - min_pt) + min_pt
                test_loss_b = L1_loss(test_p_b, label_t_press) / label_t_press.abs()
                re_L1_b = test_loss_b.mean()
                print('best model, test L1 relative error mean: {}'.format(re_L1_b.item()))
                writer.add_scalar('best model relative L1 mean error', re_L1_b, global_step=gs)

                # best model L1 loss in well condition
                re_L1_well = test_loss_b[:, well_mark].mean()
                print('best model, test L1 relative error in well grid: {}'.format(re_L1_well.item()))
                writer.add_scalar('best model relative L1 well error', re_L1_well, global_step=gs)
                gs += 1

                # adaptive sample
                if args.AS:
                # if True:
                    j = (i+1) // interval
                    # print(f'using adaptive sampling, j={j}')  # 最后一步不需要采样
                    if j <= 4:
                        X, y = adaptive_sampler.sample(layer_t, sample_num_list[j-1], criterion=L1_loss)
                        # X, y = adaptive_sampler.sample(layer_t, 10, criterion=L1_loss)
                        input_fig = torch.concat((input_fig, X), dim=0)
                        input_label = torch.concat((input_label, y), dim=0)
                        print(f'using adaptive sampling, j={j}, train_set length: {len(input_fig)}')  # 最后一步不需要采样
        
        best_model.eval()
        best_p_next = best_model(input_fig) * (max_pt - min_pt) + min_pt
        press_his.update(best_p_next.detach(), batch_id=0)

        # model[t] = best_model
        new_model_list.append(best_model)
        rand_idx = [random.randint(0, label_size - 1) for _ in range(10)]
        plot_10_figs(best_p_next[rand_idx, :], nx, ny,
                     args.fig_dir + 'train_' + args.model_name + 'sample_t_{}.png'.format(t + 1),
                     id_num=rand_idx, show=False)

        # 防止断电等意外，每一步保存一次模型，
        sequence_model = SequenceModel(new_model_list)
        torch.save(sequence_model.model_list, './model/' + args.model_name + '.pth')
        print('already save {} time step model'.format(t + 1))

        # if (t + 1) % 10 == 0:
        #     press_sample = press_his.data[-1][:, 0, :]  # [last_batch][time_step, sample_id, press_dim]
        #     plot_10_figs(press_sample[t-9: t+1], nx, ny,
        #                  args.fig_dir + 'train_' + args.model_name + '_t_{}.png'.format(t+1))
    # return sequence_model


def varify(model_list, vari_fig):
    vali_history = torch.zeros((nt, nz * ny * nz))
    sample_idx = [33, 81, 92]  # 生成数据时选取的三个样本
    vari_fig = test_fig[sample_idx]
    for t in range(nt):
        layer_t = model_list[t]
        layer_t.eval()
        p1 = layer_t(vari_fig) * (max_pt - min_pt) + min_pt
        vali_history[t] = p1.detach()

        if (t + 1) % 10 == 0:
            vari_press = vali_history.data[-1]  # [time_step, 1, press_dim]
            plot_10_figs(vari_press[t - 9: t + 1], nx, ny,
                         args.fig_dir + 'vari_' + args.model_name + '_t_{}.png'.format(t + 1))


parser = argparse.ArgumentParser()
parser.add_argument('--id', default=0, type=int,
                    help='select a gpu to train NN')
parser.add_argument('-g', '--logdir', default='./logs/label10_AS100')
parser.add_argument('--fig_dir', default='./figure/', type=str)
parser.add_argument('-t', '--dt', default=30000, type=int,
                    help='time interval of PDE model')
parser.add_argument('--max_iter', default=100, type=int,
                    help='max iters for training process of NN')
parser.add_argument('--train_data', default='./dataset/perm+press_4well.npz', type=str)
parser.add_argument('-n', '--nt', default=150, type=int,
                    help='how long will the NN model predict')
parser.add_argument('--bs', default=500, type=int, help='batch size')
parser.add_argument('--lr', default=3e-3, type=float, help='learning rate')
parser.add_argument('--interval', default=20, type=int)
parser.add_argument('-mn', '--model_name', default='label100', type=str)
parser.add_argument('-ls', '--label_size', default=100, type=int)
parser.add_argument('-wn', '--well_num', default=4, type=int)
parser.add_argument('-pt', '--pretrain', default=None,
                    type=str, help='whether use pretrain model')
# parser.add_argument('--lam', default=1, type=float)
parser.add_argument('--trans_data', default=None, type=str,
                    help='whether use computed trans data')
parser.add_argument('-RI', '--random_index', action='store_true',
                    help='whether use random index to select label data.')
parser.add_argument('--AS', action='store_true',
                    help='whether use adaptive sampling')
# parser.add_argument('-SL', '--sample_list', nargs='+', type=int)
# AS400: 50 + [100, 150, 50, 50], AS200: 40 + [60, 40, 30, 10]

args = parser.parse_args()
if __name__ == '__main__':
    ############# 参数初始化部分 ######################
    devices = check_devices()
    selected_gpu = devices[args.id]
    seed_everything(227)
    # init params and other condition
    global nt, nx, ny, nz, p_init
    nt = args.nt
    interval = args.interval
    dt = args.dt
    # pinn_size = args.pinn_size
    label_size = args.label_size
    batch_size = args.bs
    # logdir = args.logdir
    writer = SummaryWriter(comment='discriminate_data', log_dir=args.logdir)

    p_init, p_bc = 30 * 1e+6, None
    mu_o = 2e-3
    ct = 5e-8
    p_init = 30.0 * 1e6
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

    ################## 样本数据加载 ##################
    params = data_prepare(args)
    # pinn_loader, train_label_fig, train_label_press, test_fig, test_press = params['pinn_loader']
    test_fig, test_press = params['test_fig'], params['test_press']
    ################# NN params ##############
    train(args, params)
