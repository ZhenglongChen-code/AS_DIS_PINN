import os
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
from src.PDE_module import plot_10_figs
from collections import OrderedDict


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
    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.label)


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
            # if isinstance(batch_tensor, list):
            #     self.data = batch_tensor[0].unsqueeze(0)
            # else:
            #     self.data = batch_tensor.unsqueeze(0)

            self.data = [batch_tensor[i].unsqueeze(0) for i in range(self.batch_num)]

        else:
            # if isinstance(batch_tensor, list):
            #     self.data = torch.concat((self.data, batch_tensor[0].unsqueeze(0)), dim=0)
            # else:
            #     self.data = torch.concat((self.data, batch_tensor.unsqueeze(0)), dim=0)

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
parser.add_argument('-g', '--logdir', default='./logs/perm_no_label')
parser.add_argument('--fig_dir', default='./figure/', type=str)
parser.add_argument('-t', '--dt', default=30000, type=int,
                    help='time interval of PDE model')
parser.add_argument('--max_iter', default=3000, type=int,
                    help='max iters for training process of NN')
parser.add_argument('--train_data', default='./dataset/sg_1well.npz', type=str)
parser.add_argument('-n', '--nt', default=10, type=int,
                    help='how long will the NN model predict')
parser.add_argument('--bs', default=1000, type=int, help='batch size')
parser.add_argument('--lr', default=3e-3, type=float, help='learning rate')
parser.add_argument('--interval', default=300, type=int)
parser.add_argument('-mn', '--model_name', default='perm200', type=str)
parser.add_argument('-ps', '--pinn_size', default=200, type=int)
# parser.add_argument('-ls', '--label_size', default=50, type=int)

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

    # load data
    file = args.train_data
    train_permeability, test_permeability, train_label_press, test_label_press = read_file(file)
    train_permeability = train_permeability[:pinn_size]
    # save data
    writer = SummaryWriter(comment='discriminate_data', log_dir=args.logdir)

    # constant value
    trans_matrix = []

    # prepare training data
    for perm in tqdm(train_permeability, desc='data preparing'):
        mesh = MeshGrid(nx, ny, nz, perm.flatten(), mu_o, ct,
                        poro, p_init, p_bc, bhp_constant, devices=[selected_gpu])

        if len(trans_matrix) == 0:
            trans_matrix = mesh.trans_matrix.detach().flatten()
            permeability = torch.tensor(perm)
        else:
            trans_matrix = torch.vstack((trans_matrix, mesh.trans_matrix.detach().flatten()))
            permeability = torch.vstack((permeability, torch.tensor(perm)))
    if args.bs < args.pinn_size:
        batch_size = args.bs
    else:
        batch_size = args.pinn_size
    press_input = torch.tile(mesh.press, (batch_size, 1)) / p_init  # scaled press
    scaled_params = 1e-15  # scaled permeability
    train_loader = DataLoader(MyDataset(permeability / scaled_params, trans_matrix),
                              batch_size=batch_size, shuffle=False)

    neighbor_idx = mesh.neighbor_vectors  # constant value

    # label_fig = torch.tensor(train_permeability[:300]/scaled_params, dtype=torch.float64).reshape(-1, nz, nx, ny).to(selected_gpu)
    test_fig = torch.tensor(test_permeability/scaled_params, dtype=torch.float64).reshape(-1,nz,nx,ny).to(selected_gpu)
    # NN params
    criterion = nn.MSELoss()
    L1_loss = nn.L1Loss(reduction='none')
    b1 = nn.Sequential(nn.Conv2d(1, 20, kernel_size=3, stride=1, padding=1),
                       nn.BatchNorm2d(20), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))  # 1x20x20 --> 20x10x10
    b2 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
    # b3 = nn.Sequential(*resnet_block(20, 20, 2, first_block=True))
    b4 = nn.Sequential(*resnet_block(20, 40, 2, first_block=False))
    b5 = nn.Sequential(*resnet_block(40,80,2, first_block=False))

    input_size = nx * ny * nz  # the size of input tensor,
    output_size = input_size
    model = nn.Sequential(b1, GAM_Attention(20, 20),
                          b2, b4,
                          b5, GAM_Attention(80, 80), nn.Flatten(),
                          nn.Dropout(p=0.2), nn.Linear(720, output_size)).to(torch.float64).to(selected_gpu)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # scheduler = StepLR(optimizer, step_size=500, gamma=0.95)
    scheduler = CosineAnnealingLR(optimizer, T_max=2000, eta_min=4e-4)
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
        for k, trans in train_loader:
            input_fig = k.reshape(batch_size, nz, nx, ny).to(selected_gpu)
            trans = trans.to(selected_gpu)

            if t == 0:
                p_last = torch.tile(mesh.press / p_init, (batch_size, 1)).to(selected_gpu)
            else:
                # p_last = model_list[t-1](input_fig)
                # p_last = p_last.detach()
                p_last = press_history.data[batch][-1]

            max_iter = 2 * args.max_iter if t < 2 else args.max_iter
            for i in tqdm(range(max_iter), desc='training'):
                optimizer.zero_grad()
                model.train()
                # calculate PINN loss
                p_next = model(input_fig)  # size: [batch_size, 400]
                res = torch.zeros_like(p_next)
                res[:] = res[:] + (mesh.cell_volume * mesh.porosity * mesh.ct / dt) * (p_next[:] * p_init - p_last[:] * p_init)
                # for j in range(batch_size):  # 把trans_matrix 改成4个方向的tensor可以加速。
                for d in range(4):
                    res[:] = (res[:] - trans[:, d*400:(d+1)*400] *
                              (p_next[:, neighbor_idx[d]] * p_init - p_next[:] * p_init))

                res[:, 0] = res[:, 0] + mesh.PI * (p_next[:, 0] * p_init - mesh.pwf)
                loss = criterion(res, torch.zeros_like(res))
                # calculate label loss
                # label_p = model(label_fig)
                # loss2 = criterion(label_p, torch.tensor(train_label_press[:, t, :]/p_init).to(selected_gpu))
                # loss = loss1
                # loss, p_next = solve_func(p_last, trans_matrix, neighbor_idx)
                loss.backward()  # 仅在必要时使用 retain_graph=True
                optimizer.step()
                scheduler.step()
                if loss < min_loss:
                    min_loss = loss.item()
                    min_iter_time = i+1
                    best_model = copy.deepcopy(model)
                    best_p_next = p_next.clone()

                if (i + 1) % interval == 0:
                    print('\ntime step:{}, batch:{}, train step:{:d}, PINN loss{}'.format(t, batch, i + 1, loss.item()))
                    print('now, the minimum PINN loss at {}th iter, and min_loss is{}'.format(min_iter_time, min_loss))
                    model.eval()
                    test_p = model(test_fig)
                    label_t_press = torch.tensor(test_label_press[:, t, :]/p_init).to(selected_gpu)
                    test_loss = L1_loss(test_p, label_t_press) / label_t_press.abs()
                    relative_erro = test_loss.mean()
                    print('test data relative L1 error: {}'.format(relative_erro.item()))
                    writer.add_scalar('test_label_loss', relative_erro, global_step=gs)
                    writer.add_scalar('PINN_loss', loss, global_step=gs)
                    # best model L1 loss
                    test_p_b = best_model(test_fig)
                    test_loss_b = L1_loss(test_p_b, label_t_press) / label_t_press.abs()
                    re_L1_b = test_loss_b.mean()
                    print('best model test L1 relative error: {}'.format(re_L1_b.item()))
                    writer.add_scalar('best model relative L1 mean error', re_L1_b, global_step=gs)
                    gs += 1

                if loss < 1e-17:
                    print('time step:{}, batch:{}, train step:{:d}, loss{}'.format(t, batch, i + 1, loss.item()))
                    break

            # p_next[:, 0] = p_bc / p_init
            batch_press.append(best_p_next.detach())
            batch += 1

        press_history.add(batch_press)  
        model_list.append(best_model)

        if (t + 1) % 10 == 0:
            press_sample = press_history.data[-1][:, 0, :]  # [last_batch][time_step, sample_id, press_dim]
            plot_10_figs(press_sample[t-9: t+1], nx, ny,
                         args.fig_dir + 'train' + args.model_name + '_t_{}.png'.format(t+1))

    sequence_model = SequenceModel(model_list)
    torch.save(sequence_model.model_list, './model/'+args.model_name+'.pth')

    # test model
    p0 = mesh.press.detach()
    # with open('./dataset/samplesperm.txt') as f:
    #     data = f.readlines()
    #     vari_perm = list(map(float, data))
    #     vari_perm = np.array(vari_perm) * 1e-15
    vari_perm = test_permeability[0]
    vali_mesh = MeshGrid(nx, ny, nz, vari_perm.flatten(), mu_o, ct, poro,
                         p_init, p_bc, bhp_constant, [selected_gpu])
    vali_history = PressHistory()
    vari_fig = torch.tensor(vari_perm / scaled_params, dtype=torch.float64).reshape(1, 1, nx, ny).to(selected_gpu)
    for t in range(nt):
        p1 = sequence_model.predict(t+1, vari_fig)
        # p1[:, 0] = p_bc/p_init
        vali_history.add(p1.detach())

        if (t + 1) % 10 == 0:
            vari_press = vali_history.data[-1]  # [time_step, sample_id, press_dim]
            plot_10_figs(vari_press[t - 9: t + 1], nx, ny,
                         args.fig_dir + 'vari' + args.model_name + '_t_{}.png'.format(t + 1))




