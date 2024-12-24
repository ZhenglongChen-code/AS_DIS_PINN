# -*- coding: utf-8 -*-
"""
@Time ： 2024/8/8 18:17
@Auth ： ChenZL
@File ：Data.py
@IDE ：PyCharm
@Motto：ABC(Always Be Coding)
"""
from collections import OrderedDict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


class Square:
    def __init__(self, x_min=0, x_max=1, nx=50, y_min=0, y_max=1, ny=50):

        self.k_list = None
        self.type = 'space'
        self.x_min, self.x_max, self.y_min, self.y_max = x_min, x_max, y_min, y_max
        self.nx, self.ny = nx, ny

        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        self.shape = (nx, ny)

        self.index = dict({'bound_idx': np.zeros(1), 'init_idx': np.zeros(1)})
        self.X, self.Y = np.meshgrid(x, y)
        self.x_array = self.X.flatten()
        self.y_array = self.Y.flatten()
        self.point = np.vstack((self.x_array, self.y_array)).T

        # boundary condition initialize
        self.bound_val = None
        self.inner_point = None
        self.bound_point = None

        # permeability
        self.permeability = None

    def set_bound_val(self, cond='2points', val=[0, 1]):
        # boundary condition initialize
        # in this example, set left and right boundary condition
        if cond == '2lines':
            lb_idx = np.where(self.x_array == self.x_min)[0]
            rb_idx = np.where(self.x_array == self.x_max)[0]
        elif cond == '2points':
            lb_idx = np.where(np.all((self.x_array == self.x_min, self.y_array == self.y_min), axis=0))[0]
            rb_idx = np.where(np.all((self.x_array == self.x_max, self.y_array == self.y_max), axis=0))[0]

        lb_cond = np.ones(lb_idx.size) * val[0]
        rb_cond = np.ones(rb_idx.size) * val[1]
        lb, rb = self.point[lb_idx], self.point[rb_idx]

        self.bound_val = np.concatenate((lb_cond, rb_cond), axis=0)

        # bound_idx = np.where(np.any((self.x_array == x_min, self.x_array == x_max), axis=0))[0]
        bound_idx = np.append(lb_idx, rb_idx)
        inner_idx = [i for i in range(self.nx * self.ny) if i not in bound_idx]
        self.inner_point = self.point[inner_idx]
        self.bound_point = np.concatenate((lb, rb), axis=0)  # coord points of bound_x are [x=0, y] and [x=1, y]

    def set_permeability(self, k=[5, 8, 17, 20]):
        """
        :param k: a list of permeability, (e.g. [1,2,3])
        :return: a permeability field associated with the(x, y), as a simple example, we set a field
        only associated with x
        """
        n = len(k)
        self.k_list = k
        x_i = np.linspace(self.x_min, self.x_max, n + 1, endpoint=True)
        self.permeability = np.ones(len(self.point)) * k[0]
        if n > 1:
            for i in range(n):
                if i == n - 1:
                    idx = np.where(np.all((self.x_array >= x_i[i], self.x_array <= x_i[i + 1]), axis=0))[0]
                    # x <= x_max
                else:
                    idx = np.where(np.all((self.x_array >= x_i[i], self.x_array < x_i[i + 1]), axis=0))[0]
                self.permeability[idx] = k[i]

    def array2tensor(self, device_ids, embed_k=False):
        points = torch.tensor(self.point, dtype=torch.float32, requires_grad=True).cuda(device=device_ids[0])
        permeability = torch.tensor(self.permeability, dtype=torch.float32).cuda(device=device_ids[0]).reshape(-1, 1)
        bound_points = torch.tensor(self.bound_point, dtype=torch.float32,
                                    requires_grad=True).cuda(device=device_ids[0])
        bound_val = torch.tensor(self.bound_val, dtype=torch.float32).cuda(device=device_ids[0])

        if embed_k:
            points = torch.concat([points, permeability], dim=1)
            bound_k = permeability[self.index['bound_idx']]
            bound_points = torch.concat([bound_points, bound_k], dim=1)

        return points, bound_points, bound_val, permeability


class TimeSpaceDomain(Square):
    def __init__(self, t_min, t_max, nt, x_min=0, x_max=1, nx=50, y_min=0, y_max=1, ny=50,
                 cond='2points', bound_val=[0, 1], init_val=0.5):
        super().__init__(x_min=x_min, x_max=x_max, nx=nx, y_min=y_min, y_max=y_max, ny=ny)
        self.type = 'time_space'
        # data set
        self.t_min, self.t_max = t_min, t_max
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        t = np.linspace(t_min, t_max, nt)
        self.shape = (nx, ny, nt)
        self.X, self.Y, self.T = np.meshgrid(x, y, t, indexing='ij')
        self.x_array = self.X.flatten()
        self.y_array = self.Y.flatten()
        self.t_array = self.T.flatten()

        self.point = np.vstack((self.x_array, self.y_array, self.t_array)).T

        self.index = dict({'bound_idx': np.zeros(1), 'init_idx': np.zeros(1)})

        # bound and initial conditions
        self.bound_val = None
        self.set_bound_val(cond=cond, val=bound_val)

        self.init_point = None
        self.init_val = None
        self.set_initial_val(init_val)

        # self.set_permeability()

    def set_bound_val(self, cond='2points', val=[0, 1]):
        if cond == '2lines':
            lb_idx = np.where(self.x_array == self.x_min)[0]
            rb_idx = np.where(self.x_array == self.x_max)[0]
        elif cond == '2points':
            lb_idx = np.where(np.all((self.x_array == self.x_min, self.y_array == self.y_min), axis=0))[0]
            rb_idx = np.where(np.all((self.x_array == self.x_max, self.y_array == self.y_max), axis=0))[0]

        lb_cond = np.ones(lb_idx.size) * val[0]
        rb_cond = np.ones(rb_idx.size) * val[1]

        lb, rb = self.point[lb_idx], self.point[rb_idx]

        self.bound_val = np.concatenate((lb_cond, rb_cond), axis=0).reshape(-1, 1)
        bound_idx = np.append(lb_idx, rb_idx)
        self.index['bound_idx'] = bound_idx
        inner_idx = [i for i in range(self.nx * self.ny) if i not in bound_idx]
        self.inner_point = self.point[inner_idx]
        self.bound_point = np.concatenate((lb, rb), axis=0)  # coord of boundary points

    def set_initial_val(self, val):
        init_idx = np.where(self.t_array == self.t_min)[0]
        init_idx = [i for i in init_idx if i not in self.index['bound_idx']]
        self.index['init_idx'] = init_idx
        self.init_val = np.ones((len(init_idx), 1)) * val
        self.init_point = self.point[init_idx]

    def array2tensor(self, device_ids, embed_k=False):
        points = torch.tensor(self.point, dtype=torch.float32, requires_grad=True).cuda(device=device_ids[0])
        bound_points = torch.tensor(self.bound_point, dtype=torch.float32, requires_grad=True).cuda(
            device=device_ids[0])
        bound_val = torch.tensor(self.bound_val, dtype=torch.float32).cuda(device=device_ids[0])
        init_point = torch.tensor(self.init_point, dtype=torch.float32, requires_grad=True).cuda(device=device_ids[0])
        init_val = torch.tensor(self.init_val, dtype=torch.float32).cuda(device=device_ids[0])
        permeability = torch.tensor(self.permeability, dtype=torch.float32).cuda(device=device_ids[0]).reshape(-1, 1)

        if embed_k:
            points = torch.concat([points, permeability], dim=1)
            bound_k = permeability[self.index['bound_idx']]
            bound_points = torch.concat([bound_points, bound_k], dim=1)
            init_k = permeability[self.index['init_idx']]
            init_point = torch.concat([init_point, init_k], dim=1)

        return points, bound_points, bound_val, init_point, init_val, permeability


def visualize_k(domain):
    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
    k = domain.permeability.reshape(domain.shape)

    if len(domain.shape) > 2:
        c = ax.pcolormesh(domain.X[:, :, 0], domain.Y[:, :, 0], k[:, :, 0],
                          shading='auto', cmap=plt.cm.jet)
    else:
        c = ax.pcolormesh(domain.X, domain.Y, k,
                          shading='auto', cmap=plt.cm.jet)
    # points = np.random.rand(20,2)
    # ax.plot(points[:,0], points[:,1], 'kx', markersize=5, clip_on=False, alpha=1.0)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(domain.x_min, domain.x_max)
    ax.set_ylim(domain.y_min, domain.y_max)
    ax.set_title('permeability')
    fig.colorbar(c, ax=ax)
    plt.show()


# test cases

# domain = Square(nx=10, ny=10)
# domain.set_bound_val()
# domain.set_permeability()
# visualize_k(domain)

# domain = TimeSpaceDomain(t_min=0, t_max=1, nt=10, x_min=0, x_max=1, nx=40, y_min=0, y_max=1, ny=40)
# domain.set_permeability()
# visualize_k(domain)

class PressHistory:
    def __init__(self):
        self.data = []
        self.batch_num = 1

    def add(self, batch_tensor):
        """
        actually we only need store 1 batch press data as an example,
        to be simple, you can just add last batch press tensor.
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


class SequenceModel(nn.Module):
    def __init__(self, model_list):
        super(SequenceModel, self).__init__()
        self.time_step = len(model_list)
        self.model_list = model_list
        layer_list = []
        for t in range(self.time_step):
            layer_list.append(('step_%d' % t, model_list[t]))

        self.layer = nn.Sequential(OrderedDict(layer_list))

    def predict(self, time_step, input_fig):
        press_out = self.model_list[time_step - 1](input_fig)
        return press_out


def write_into_vtk(mesh, file_path: str):
    node_list, cell_list = mesh.node_list, mesh.cell_list
    node_num, cell_num = len(node_list), len(cell_list)
    with open(file_path, 'w') as f:
        f.write("# vtk DataFile Version 2.0\n")
        f.write("Unstructured Grid\n")
        f.write("ASCII\n")
        f.write('DATASET UNSTRUCTURED_GRID\n')
        f.write('\n')

        # write POINTS
        f.write('POINTS {:d} double\n'.format(node_num))
        for node in node_list:
            x, y, z = node.coord
            f.write(f'{x:.3f} {y:.3f} {z:.3f}\n')

        f.write('\n')

        # write CELL
        f.write("CELLS {:d} {:d}\n".format(cell_num, 9 * cell_num))

        for i, cell in enumerate(cell_list):
            f.write(f'{len(cell.vertices)}')
            for j in cell.vertices:
                f.write(f' {j:d}')

            f.write('\n')

        f.write('\n')

        # write CELL_TYPES
        f.write("CELL_TYPES {:d}\n".format(cell_num))
        for i in range(cell_num):
            f.write("11\n")
        f.write('\n')

        # write CELL DATA
        f.write("CELL_DATA {:d}\n".format(cell_num))
        f.write("SCALARS Permeability_mD double\n")
        f.write("LOOKUP_TABLE default\n")
        for cell in cell_list:
            f.write(f'{cell.kx * 1e15:.3f}\n')
        f.write('\n')

        f.write("SCALARS Pressure double\n")
        f.write("LOOKUP_TABLE default\n")
        for cell in cell_list:
            f.write(f'{cell.press/10**6:.3f}\n')
        f.write('\n')

        # f.write("SCALARS Saturation double\n")
        # f.write("LOOKUP_TABLE Sw\n")
        # for cell in cell_list:
        #     f.write(f"{cell.Sw:.3f}\n")


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
    def __init__(self, X):
        self.data = X
        # self.label = y

    def __getitem__(self, idx):
        return self.data[idx]

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


