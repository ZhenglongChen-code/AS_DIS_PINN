import argparse
import torch
import numpy as np
from src.PDE_module import plot_10_figs, check_devices
from src.Mesh import *
from src.Data import PressHistory
from tqdm import tqdm
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from src.plot import show_filed


def read_file(filepath):
    train_data = np.load(filepath)
    train_permeability, test_permeability = train_data['train_permeability'], train_data['test_permeability']
    # train_label_press, test_label_press = train_data['train_label_press'], train_data['test_label_press']
    train_label_press, test_label_press = train_data['train_press'], train_data['test_press']
    return train_permeability, test_permeability, train_label_press, test_label_press


def compute_qwt(args, net_list):
    qwt = np.zeros((nt, len(test_fig), args.well_num))
    for t in tqdm(range(nt), desc='compute qwt'):
        layer_t = net_list[t].to(selected_gpu)
        layer_t.eval()
        p_next = layer_t(test_fig) * (max_pt - min_pt) + min_pt

        # compute qwt
        qw_c = (p_next[:, well_mark] - pwf)  # shape: [420, 4]; mean:
        qwt[t] = qw_c.detach().cpu().numpy()

    qwt_mean = qwt.mean(axis=1)
    qwt_sum = qwt_mean.cumsum(axis=0)
    for j in range(args.well_num):
        colum_name = args.model_name + '_' + str(j)
        if colum_name not in qwt_mean_sum.columns:
            qwt_mean_sum.insert(0, colum_name, qwt_sum[:, j])
        else:
            qwt_mean_sum[colum_name] = qwt_sum[:, j]

    qwt_mean_sum.to_csv('./dataset/{}well_nt{}_qwt_sum.txt'.format(args.well_num, args.nt),
                        sep='\t', index=False, header=True)
    sns.relplot(data=qwt_mean_sum, kind='line')
    plt.xlabel('time step')
    plt.ylabel('accumulative qwt')
    plt.legend(title='well number')
    plt.savefig('./' + args.model_name + '_qwt_mean_sum.png')
    plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--id', default=0, type=int,
                    help='select a gpu to train NN')
parser.add_argument('--fig_dir', default='./varify/', type=str)  #
parser.add_argument('-t', '--dt', default=30000, type=int,
                    help='time interval of PDE model')
parser.add_argument('--testdata', default='./dataset/testdata.pth', type=str)
parser.add_argument('-n', '--nt', default=150, type=int,  #
                    help='how long will the NN model predict')
parser.add_argument('-mn', '--model_name', default='label800_s_4well')  # pinn1600+label800_4well
parser.add_argument('-wn', '--well_num', default=4, type=int)
parser.add_argument('-st', '--start', default=0, type=int)
args = parser.parse_args()

if __name__ == '__main__':

    devices = check_devices()
    selected_gpu = devices[args.id]

    # init and condition
    global nt, pwf, test_fig, test_press, max_pt, min_pt
    p_init, p_bc = 30 * 1e+6, None  # 没有边界条件
    mu_o = 2e-3
    ct = 5e-8
    p_init = 30.0 * 1e6
    p_scale = 1e6
    nt = args.nt
    dt = args.dt
    # max_iter = args.max_iter
    # qw = 0.0005
    chuk = 5e-15
    poro = 0.1
    rw = 0.05
    SS = 3
    length = 3000
    cs = ct * 3.14*rw*rw*length
    bhp_constant = 20*1e6
    pwf = bhp_constant
    nx, ny, nz = 20, 20, 1  # x, y, z 方向的cell数目, 这个方向顶点个数要比cell数目多一个
    if args.well_num == 1:
        well_mark = [0]
    else:
        well_mark = [0, 19, 380, 399]

    # load data and model params
    test_data = torch.load(args.testdata)  # 使用预先保存的 T_ij数据
    test_fig, test_press = test_data.values()
    config = torch.load('./dataset/config_' + 'pinn1600+label800_4well' + '.pth')
    max_pt, min_pt, scaled_params, neighbor_idx = config.values()
    model_name = args.model_name
    model = torch.load('./model/' + model_name + '.pth', map_location=lambda storage, loc: storage)
    print('The model has {} time-layers'.format(len(model)))

    origin_data = pd.read_csv('dataset/{:d}well_press-dif_nt150_data_re.txt'.format(args.well_num), header=0, sep='\t')
    # origin_data = pd.read_csv('dataset/{:d}well_nt150_data_re.txt'.format(args.well_num), header=0, sep='\t')
    global qwt_mean_sum
    qwt_mean_sum = pd.read_csv('./dataset/{}well_nt{}_qwt_sum.txt'.format(args.well_num, args.nt), header=0, sep='\t')
    qwt = np.zeros((nt, len(test_fig), args.well_num))

    # data record
    press_his = test_press
    model_his = PressHistory()
    well_error = []
    mean_error = []
    # criterion = torch.nn.MSELoss()
    L1Loss = torch.nn.L1Loss(reduction='none')  # return a vector

    # compute relative L1 error
    for t in tqdm(range(0, nt)):
        layer_t = model[t].to(selected_gpu)
        if t == 0:
            p_last = torch.ones((len(test_fig), 400)).to(torch.float64).to(selected_gpu)

        # p_next = layer_t(test_fig) * p_scale
        layer_t.eval()
        p_next = layer_t(test_fig) * (max_pt - min_pt) + min_pt

        label_p = torch.tensor(press_his[:, t, :], dtype=torch.float64).to(selected_gpu)
        all_sample_L1 = L1Loss(p_next, label_p) / (p_init - label_p.abs())
        # all_sample_L1 = L1Loss(p_next, label_p) / label_p.abs()
        # print('\ntime: {}, mean L1 error: {}, well L1 error: {}'.format(t, all_sample_L1.mean(),
        #                                                                 all_sample_L1[:, well_mark].mean()))

        for i, press in enumerate(p_next):
            L1_error = L1Loss(press, label_p[i]) / (p_init - label_p[i].abs())
            # L1_error = L1Loss(press, label_p[i]) / label_p[i].abs()
            # 生产井相对误差
            well_relative_error = L1_error[well_mark]
            # 全部位置相对误差
            mean_relative_error = L1_error.mean()

            well_error.append(well_relative_error.mean().item())
            mean_error.append(mean_relative_error.item())
        # print('relative error: {}'.format(relative_error))

        # if (t+1)%10 == 0:
        #     plot_10_figs(press_his[t-9:t+1], 20, 20, args.fig_dir+'test_FVM_t{}.png'.format(t+1))
        #     plot_10_figs(model_his.data[0][t-9:t+1] * p_init, 20, 20, args.fig_dir+'test_model_t{}.png'.format(t+1))

    # compute_qwt(args, model)

# show_filed(p_next[10, :], nx, ny, args.model_name + '_sample1.png', fig_path='./varify/')
# show_filed(p_next[150, :], nx, ny, args.model_name + '_sample2.png', fig_path='./varify/')
# show_filed(p_next[30, :], nx, ny, args.model_name + '_sample3.png', fig_path='./varify/')
# show_filed(p_next[200, :], nx, ny, args.model_name + '_sample4.png', fig_path='./varify/')

well_error = np.abs(np.array(well_error))
mean_error = np.abs(np.array(mean_error))
print('well mean error {}, all block mean error {}'.format(well_error.mean(), mean_error.mean()))
log_well_error = np.log10(well_error)
log_mean_error = np.log10(mean_error)
data = {'log_well_error': log_well_error, 'log_mean_error': log_mean_error}
re_error = pd.DataFrame(data)

# qwt_df = pd.DataFrame({})


plt.figure()
sns.set(style="ticks", palette="muted")
# sns.displot(data=re_error, x='log_well_error', kde=True, bins=50)
sns.kdeplot(data=re_error, x='log_well_error',color='b', label='model predict well relative error')
sns.kdeplot(data=origin_data, x='log well relative error', color='r', label='FVM well relative error')
plt.title('varify ' + args.model_name + ' well relative error')
plt.legend()
# plt.title('不同分布的概率密度函数')
plt.xlabel('X')
plt.ylabel('probability density')
plt.savefig('./varify/' + args.model_name + 'well.png', dpi=400, bbox_inches='tight')
plt.show()

plt.figure()
sns.set(style="ticks", palette="muted")
# sns.displot(data=re_error, x='log_mean_error', kde=True, bins=50)
sns.kdeplot(data=re_error, x='log_mean_error',color='b', label='model predict mean relative error')
sns.kdeplot(data=origin_data, x='log mean relative error', color='r', label='FVM mean relative error')
plt.title('varify ' + args.model_name + ' mean relative error')
plt.legend()
# plt.title('不同分布的概率密度函数')
plt.xlabel('X')
plt.ylabel('probability density')
plt.savefig('./varify/' + args.model_name + 'mean.png', dpi=400, bbox_inches='tight')
plt.show()

re_error.to_csv('./dataset/' + args.model_name + '_press-dif_nt{}.txt'.format(nt), sep='\t', index=False, header=True)
# re_error.to_csv('./dataset/' + args.model_name + '_nt{}.txt'.format(nt), sep='\t', index=False, header=True)