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
parser.add_argument('-g', '--logdir', default='./logs/sg_pinn')
parser.add_argument('--fig_dir', default='./figure/', type=str)
parser.add_argument('-t', '--dt', default=30000, type=int,
                    help='time interval of PDE model')
parser.add_argument('--max_iter', default=500, type=int,
                    help='max iters for training process of NN')
parser.add_argument('--train_data', default='./dataset/sg_1well.npz', type=str)
parser.add_argument('-n', '--nt', default=10, type=int,
                    help='how long will the NN model predict')
parser.add_argument('--bs', default=1000, type=int, help='batch size')
parser.add_argument('--lr', default=1e-2, type=float, help='learning rate')
parser.add_argument('--interval', default=200, type=int)
parser.add_argument('-mn', '--model_name', default='trans_pinn_1000_1well', type=str)
parser.add_argument('-ps', '--pinn_size', default=1000, type=int)
# parser.add_argument('-ls', '--label_size', default=50, type=int)
parser.add_argument('-wn', '--well_num', default=1, type=int)

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
    if args.well_num == 1:
        well_mark = [0]  # 源的位置
    else:
        well_mark = [0, 19, 380, 399]

    if args.bs < args.pinn_size:
        batch_size = args.bs
    else:
        batch_size = args.pinn_size

    # 1. load data
    file = args.train_data
    train_permeability, test_permeability, train_label_press, test_label_press = read_file(file)

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
    train_trans_matrix = compute_trans(train_permeability[:pinn_size], desc='compute train trans_matrix')
    train_trans_matrix = train_trans_matrix.reshape(-1, 4, ny, nx)
    # press_input = torch.tile(mesh.press, (batch_size, 1)) / p_init  # scale press into [0, 1]
    scaled_params = 1e-12  # scaled tans_matrix into [0, 1000]
    # permeability = torch.tensor(train_permeability_scaled, dtype=torch.float64)
    train_loader = DataLoader(MyDataset(train_trans_matrix / scaled_params),
                              batch_size=batch_size, shuffle=False)

    # load pretrain model
    model_name = 'Pretrain_trans1000_1well'
    # model = torch.load('./model/' + model_name + '.pth', map_location=lambda storage, loc: storage)
    # layer_t = model[-1].to(selected_gpu)
    # torch.save(layer_t, './model/Pretrain_trans1000_1well.pth')
    model = torch.load('./model/' + model_name + '.pth', map_location=lambda storage, loc: storage)
    model = model.to(selected_gpu)

    # predict
    input_fig = (train_trans_matrix / scaled_params).reshape(-1, 4, ny, nx).to(selected_gpu)
    model.eval()
    p_next = model(input_fig)

    show_filed(p_next[0].detach(), nx, ny, 'Pretrain_press.png', show=True)

