import argparse
import numpy as np
from src.Net_structure import check_devices
from src.plot import show_filed, plot_10_figs
from src.Mesh import *
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.Data import write_into_vtk

parser = argparse.ArgumentParser()
parser.add_argument('--id', default=0, type=int,
                    help='select a gpu to train NN')
# parser.add_argument('-l', '--logdir', default='./logs/perm+label', help='-h: j=history')
parser.add_argument('--fig_dir', default='./figure/', type=str)
parser.add_argument('-t', '--dt', default=30000, type=int,
                    help='time interval of PDE model')
# parser.add_argument('--train_data', default='./dataset/perm_ls.npz', type=str)
parser.add_argument('-n', '--nt', default=150, type=int,
                    help='how long will the NN model predict')
parser.add_argument('-fn', '--file_name', default='perm+press_4well.npz',
                    help='save  well simulation press label')
parser.add_argument('-wn', '--well_num', default=4, type=int)

args = parser.parse_args()

if __name__ == '__main__':
    devices = check_devices()
    selected_gpu = devices[args.id]
    # seed_everything(227)
    # init params and other condition
    nt = args.nt
    dt = args.dt

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
    well_mark = [0]
    print('well num: {}'.format(args.well_num))
    if args.well_num > 1:
        well_mark = [0, 19, 380, 399]

    permeability_untransformed = np.loadtxt('./dataset/permeability-sgsim20x20-2100samples.txt', skiprows=0)  # 每列是一个样本
    permeability = 2 ** permeability_untransformed * 0.1 * 1e-15

    # permeability = permeability[:3, :]
    press_his = np.zeros((len(permeability), nt, nz*ny*nx))
    for i, perm in enumerate(tqdm(permeability)):
        mesh = MeshGrid(nx, ny, nz, permeability[i].flatten(), mu_o, ct,
                        poro, p_init, p_bc, bhp_constant, devices=[selected_gpu], well_mark=well_mark)

        press, _ = mesh.solve_dynamic_press(dt, nt, np.ones(mesh.ncell) * p_init)
        press_his[i] = press

    X_train, X_test, y_train, y_test = train_test_split(permeability, press_his, test_size=0.2)
    # 保存perm 和 pressure的对应数据
    np.savez('./dataset/' + args.file_name, train_permeability=X_train, train_press=y_train,
             test_permeability=X_test, test_press=y_test)

    for i in range(10):  # 检查一下前200个样本中压力差别, 后续找差别最大的样本来验证模型的预测
        plot_10_figs(y_test[i*10:(i+1)*10, -1, :], nx, ny,
                     args.fig_dir + 'FVM_test_sample_{}.png'.format((i + 1)*10), show=True)

    for i in [33, 81, 92]:
        press = y_test[i]
        mesh.update_cell(press[-1, :])
        write_into_vtk(mesh, f'./vtk/FVM_test_sample_{i}_{args.well_num}well.vtk')
        show_filed(press[-1, :], nx, ny, f'FVM_test_sample_{i}_{args.well_num}well.png', './figure/', True)



