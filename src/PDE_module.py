import copy
from collections import OrderedDict
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from tqdm import trange
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.mixture import GaussianMixture
"""
This file provides two ways for design a PINN model, one is to create a PINN class and the other is to create several
functions and combine these functions and dataset together. The first method is more efficient and easy to use, but the
second way is more flexible.

"""


def select_point(x_range, y_range, z):
    idx1 = np.where(np.all([x_range[0] <= z[:, 0], z[:, 0] <= x_range[1]], axis=0))[0]
    idx2 = np.where(np.all([y_range[0] <= z[:, 1], z[:, 1] <= y_range[1]], axis=0))[0]

    idx = list(set(idx1) & set(idx2))

    return z[idx]


class DNN(torch.nn.Module):
    def __init__(self, layers, activation=nn.Tanh):
        """
        :param layers: size of each layer of nn, such as [2, 5, 5, 2]
        :param activation: activation function, default: Tanh
        """
        super(DNN, self).__init__()
        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Tanh

        layer_list = list()
        for i in range(self.depth - 1):  # last layer do not need activation layer
            layer_list.append(
                ('hidden_layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1]))
            )
            layer_list.append(('activation_%d' % i, self.activation()))
            layer_list.append(('batch_norm_%d' % i, torch.nn.BatchNorm1d(layers[i+1])))

        layer_list.append(
            ('hidden_layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


def check_devices():
    gpu_count = torch.cuda.device_count()
    devices = [torch.device(f'cuda:{i}') for i in range(gpu_count)]
    print('gpu num is: {}'.format(gpu_count))
    return devices


def pde_loss(X: torch.Tensor, exact, solver, criterion=nn.MSELoss()):
    p_pre = solver(X)
    loss = criterion(exact, p_pre)
    return loss


def pde_residual(X: torch.Tensor, output: torch.Tensor, k) -> torch.Tensor:
    """
    compute pde residual function, which will be used in the PINN class, it should be rewritten for a new problem.
    :param X: input point data [x,y,t]
    :param output:
    :return: e.g. pde: L(U) = f, return res = L(U) - f, res should converge to 0, L is an operator related to pde.
    """
    p_x = torch.autograd.grad(output, X, grad_outputs=torch.ones_like(output),
                              retain_graph=True,
                              create_graph=True)[0][:, 0]
    p_y = torch.autograd.grad(output, X, grad_outputs=torch.ones_like(output),
                              retain_graph=True,
                              create_graph=True)[0][:, 1]
    p_xx = torch.autograd.grad(p_x, X, grad_outputs=torch.ones_like(p_x),
                               retain_graph=True,
                               create_graph=True)[0][:, 0]
    p_yy = torch.autograd.grad(p_y, X, grad_outputs=torch.ones_like(p_y),
                               retain_graph=True,
                               create_graph=True)[0][:, 1]
    # k = X[:, -1]
    return k * (p_xx + p_yy)


def train(net, domain, ped_res_func, optimizer,
          criterion, scheduler, max_iter, device_ids, interval=1000,
          logdir='./logs/exp_0', embed_k=False):
    # setup tensorboard writer
    writer = SummaryWriter(logdir)
    if domain.type == 'time_space':
        points, bound_points, bound_val, init_points, init_val, permeability = domain.array2tensor(device_ids, embed_k)
    else:
        points, bound_points, bound_val, permeability = domain.array2tensor(device_ids, embed_k)

    for epoch in trange(max_iter, desc='training'):
        net.train()
        optimizer.zero_grad()
        # compute boundary loss
        loss_bound = pde_loss(bound_points, bound_val, solver=net)
        # compute pde residual in hole area
        outputs = net(points)
        res = ped_res_func(points, outputs)
        res_ave = criterion(res, torch.zeros_like(res))
        loss = loss_bound + res_ave
        writer.add_scalars('loss', {'loss_bound': loss_bound.detach(), 'residual': res_ave.detach()}, epoch + 1)
        if domain.type == 'time_space':
            loss_init = pde_loss(init_points, init_val, solver=net)
            writer.add_scalar('loss_init', loss_init, epoch + 1)
            loss += loss_init

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % interval == 0:
            print('train {:d} steps, train loss: {:.9f}'.format(epoch + 1, loss.detach().cpu()))
            if domain.type == 'space':
                fig = visualize(points, net, domain.shape)
            else:
                fig = visualize_t(points, net, domain.shape)
            writer.add_figure('solution_press', fig, epoch + 1)


def visualize(X, solver, shape):
    out = solver(X)
    out = out.detach().cpu().numpy()

    fig = plt.figure(dpi=300, figsize=(4, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    x, y = X[:, 0], X[:, 1]
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

    im = ax.imshow(out.reshape(shape), cmap='viridis',
                   extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(y.min(), y.max())
    ax.plot(x, y, 'kx', markersize=5, clip_on=False, alpha=1.0)
    ax.set_title('press_solution')
    cbar = fig.colorbar(im, ax=ax, orientation='vertical', extend='both',
                        ticks=[0, 0.25, 0.5, 0.75, 1], format='%.2f', label='Press Values')
    # 设置 colorbar 的刻度标签大小
    cbar.ax.tick_params(labelsize=10)
    # plt.show()
    return fig


def visualize_t(X, solver, shape):
    out = solver(X).detach().cpu().numpy()
    # shape = self.domain.shape
    out = out.reshape(shape)

    fig = plt.figure(dpi=600, figsize=(15, 15))
    axs = [fig.add_subplot(3, 4, i + 1) for i in range(10)]

    x, y = X[:, 0], X[:, 1]

    if isinstance(X, torch.Tensor):
        x = x.detach().cpu().numpy()
        y = y.detach().cpu().numpy()

    for i in range(10):
        axs[i].set_xlabel('X')
        axs[i].set_ylabel('Y')

        # rand_ind = [np.random.randint(0, len(self.bound_point)) for _ in range(20)]

        im = axs[i].imshow(out[:, :, i], cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
        axs[i].set_xlim(x.min(), x.max())
        axs[i].set_ylim(y.min(), y.max())
        # axs[i].plot(x, y, 'kx', markersize=5, clip_on=False, alpha=1.0)
        axs[i].set_title('press_t{}'.format(i))
        if i == 9:
            cbar = fig.colorbar(im, ax=axs[i], orientation='vertical', extend='both',
                                ticks=[0, 0.25, 0.5, 0.75, 1], format='%.2f', label='Press Values')
            # 设置 colorbar 的刻度标签大小
            cbar.ax.tick_params(labelsize=10)
    # plt.show()
    return fig


class PINN():
    def __init__(self, solver_layers, domain, device_ids, log_dir='./logs', 
                 pde_func=pde_residual, lr=1e-2):
        """
        :param solver_layers: pde solver layers;
        :param domain: data point domain;
        :param device_ids:
        """
        # time & space params
        self.domain = domain
        self.device_ids = device_ids

        # setup init condition
        if self.domain.type == 'time_space':
            self.X, self.bound_point, self.bound_val, self.init_point, self.init_val, self.k = domain.array2tensor(
                device_ids)
        else:
            self.X, self.bound_point, self.bound_val, self.k = domain.array2tensor(device_ids)
            self.init_point = None
            self.init_val = None

        self.original_X = self.X

        # PINN params
        self.solver = DNN(solver_layers).cuda(device=device_ids[0])
        if len(device_ids) > 1:
            self.solver = nn.DataParallel(self.solver, device_ids=device_ids)

        self.optimizer = torch.optim.Adam(self.solver.parameters(), lr=lr)
        # self.optimizer = torch.optim.LBFGS(self.solver.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.95)
        self.criterion = nn.MSELoss()

        # adaptive sample parameters
        # self.n_components = list(range(2, 8))
        self.samplers = [GaussianMixture(n_components=i, covariance_type='full', max_iter=2000) for i in list(range(4, 5))]
        self.generate_num = len(self.X) // 10

        # pde parameters
        self.residual_func = pde_func

        # other params
        self.min_loss = 1e+5
        self.best_solver = None
        self.writer = SummaryWriter(comment='PINNS_data', log_dir=log_dir)
        self.train_step = 0

    def data_update(self, add_point):
        # if not isinstance(add_point, torch.Tensor):
        #     add_point = add_point.detach()

        k = self.domain.k_list
        n = len(k)
        x_i = np.linspace(self.domain.x_min, self.domain.x_max, n + 1, endpoint=True)
        new_permeability = np.ones(len(add_point)) * k[0]
        if n > 1:
            for i in range(n):
                if i == n - 1:
                    idx = np.where(np.all((add_point[:, 0] >= x_i[i], add_point[:, 0] <= x_i[i + 1]), axis=0))[0]
                    # x <= x_max
                else:
                    idx = np.where(np.all((add_point[:, 0] >= x_i[i], add_point[:, 0] < x_i[i + 1]), axis=0))[0]
                new_permeability[idx] = k[i]
        new_permeability = torch.tensor(new_permeability, dtype=torch.float32).cuda(device=self.device_ids[0]).reshape(-1, 1)
        add_point = torch.tensor(add_point, dtype=torch.float32).cuda(device=self.device_ids[0])
        self.X = torch.vstack((self.X, add_point))
        self.k = torch.vstack((self.k, new_permeability))

    def train_loss(self):
        """
        :return: loss_bound + residual
        """
        bound_pre = self.solver(self.bound_point)
        loss_bound = self.criterion(bound_pre, self.bound_val.reshape(bound_pre.shape))
        loss_init = 0
        if self.init_point is not None:
            init_pre = self.solver(self.init_point)
            loss_init = self.criterion(init_pre, self.init_val.reshape(init_pre.shape))
            self.writer.add_scalar('init_loss', scalar_value=loss_init.detach(), global_step=self.train_step)

        res_vec = self.get_pde_residual()
        residual = self.criterion(res_vec, torch.zeros_like(res_vec))
        # self.loss_history.append([loss_bound.detach().cpu(), residual.detach().cpu()])
        self.train_step += 1
        # self.writer.add_scalar(tag='bound_loss', scalar_value=loss_bound.detach(), global_step=self.train_step)
        # self.writer.add_scalar(tag='residual', scalar_value=residual.detach(), global_step=self.train_step)
        self.writer.add_scalars('Loss', {'bound_loss': loss_bound, 'residual': residual}, self.train_step)
        # bound loss and init loss are difficult to decrease
        return loss_bound + residual + loss_init, res_vec

    def get_pde_residual(self):
        X = self.X
        output = self.solver(X)
        res = self.residual_func(X, output, self.k)
        return res

    def train_solver(self, max_iter=1000, interval=100):
        self.solver.train()
        for epoch in trange(max_iter, desc='training'):
            self.optimizer.zero_grad()
            # using LBFGS optimizer
            # def closure():
            #     loss = self.train_loss()
            #     loss.backward()

            #     self.writer.add_scalar(tag='sum_loss', scalar_value=loss.detach(), global_step=self.train_step)
            #     if loss < self.min_loss:
            #         self.best_solver = copy.deepcopy(self.solver)

            #     if (epoch + 1) % interval == 0:
            #         print('train {:d} steps, train loss: {}'.format(epoch + 1, loss.detach().cpu()))
            #         fig = self.visualize_t()
            #         self.writer.add_figure(tag='solution', figure=fig, global_step=(epoch + 1))
    
            #     return loss
            
            # using Adam optimizer
            loss, res_vec = self.train_loss()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            self.writer.add_scalar(tag='sum_loss', scalar_value=loss.detach(), global_step=self.train_step)

            if loss < self.min_loss:
                self.best_solver = copy.deepcopy(self.solver)

            if (epoch + 1) % interval == 0:
                print('train {:d} steps, train loss: {}'.format(epoch + 1, loss.detach().cpu()))

                fig = self.visualize_t()
                self.writer.add_figure(tag='solution', figure=fig, global_step=(epoch + 1))
                fig2 = self.visualize_res(res_vec.abs())  # plot abs residual
                self.writer.add_figure(tag='residual', figure=fig2, global_step=(epoch + 1))

                # using adaptive sample:
                for sampler in self.samplers:
                    # res_vec = self.get_pde_residual().detach().cpu().numpy()
                    idx_res = torch.where(res_vec > 0.5 * res_vec.max())[0]
                    sampler.fit(self.X[idx_res].detach().cpu().numpy())
                    Z, Z_label = sampler.sample(self.generate_num)
                    x_range = [self.domain.x_min, self.domain.x_max]
                    y_range = [self.domain.y_min, self.domain.y_max]
                    X_new = select_point(x_range, y_range, Z)
                    while len(X_new) < self.generate_num//2:
                        Z, Z_label = sampler.sample(self.generate_num)
                        X_new = np.vstack((X_new, select_point(x_range, y_range, Z)))
                    fig3 = self.visualize_res(res_vec.abs(), add_point=X_new)  # plot abs residual and add_point
                    self.writer.add_figure(tag='residual and add_point', figure=fig3, global_step=(epoch + 1))
                    # update self.X
                    self.data_update(add_point=X_new)

        self.writer.add_graph(self.solver, input_to_model=self.X)
        self.writer.close()


    @torch.no_grad()
    def predict(self, point=None, c='last'):
        if c == 'last':
            solver = self.solver
        elif c == 'best':
            solver = self.best_solver

        solver.eval()
        if point is not None:
            if isinstance(point, np.ndarray):
                point = torch.tensor(point, dtype=torch.float32, requires_grad=False, device=self.device)

            out_pre = solver(point)
            return out_pre.detach().cpu().numpy()
        else:
            # using default point
            out_pre = solver(self.X)
            return out_pre.detach().cpu().numpy()
    
    def visualize(self):
        out = self.predict(point=self.original_X)
        shape = self.domain.shape
        fig = plt.figure(dpi=300, figsize=(4, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # rand_ind = [np.random.randint(0, len(self.bound_point)) for _ in range(20)]
        x, y = self.domain.bound_point[:, 0], self.domain.bound_point[:, 1]
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
        im = ax.imshow(out.reshape(shape), cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.plot(x, y, 'kx', markersize=5, clip_on=False, alpha=1.0)
        ax.set_title('press_solution')
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', extend='both',
                            ticks=[0, 0.25, 0.5, 0.75, 1], format='%.2f', label='Press Values')
        # 设置 colorbar 的刻度标签大小
        cbar.ax.tick_params(labelsize=10)
        # plt.show()
        return fig

    def visualize_t(self):
        out = self.predict(point=self.original_X)
        out = out[list(range(len(self.original_X)))]
        shape = self.domain.shape
        out = out.reshape(shape)
        nt = shape[-1]

        fig = plt.figure(dpi=600, figsize=(15, 15))
        axs = [fig.add_subplot(3, 4, i + 1) for i in range(nt)]

        x, y = self.domain.bound_point[:, 0], self.domain.bound_point[:, 1]

        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

        for i in range(nt):
            axs[i].set_xlabel('X')
            axs[i].set_ylabel('Y')

            # rand_ind = [np.random.randint(0, len(self.bound_point)) for _ in range(20)]
            value = out[:, :, i]
            im = axs[i].imshow(value, cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()],
                               origin='lower')

            axs[i].set_xlim(x.min(), x.max())
            axs[i].set_ylim(y.min(), y.max())
            axs[i].plot(x, y, 'kx', markersize=5, clip_on=False, alpha=1.0)
            axs[i].set_title('press_t{}'.format(i))
            if i == 9:
                cbar = fig.colorbar(im, ax=axs[i], orientation='vertical', extend='both',
                                    ticks=np.linspace(value.min(), value.max(), 5, endpoint=True), format='%.2f',
                                    label='Press Values')
                # 设置 colorbar 的刻度标签大小
                cbar.ax.tick_params(labelsize=10)
        # plt.show()
        return fig

    def visualize_res(self, res_vector, add_point=None):
        out = res_vector.detach().cpu().numpy()
        out = out[list(range(len(self.original_X)))]
        res_min, res_max = np.min(out), np.max(out)
        shape = self.domain.shape
        # print('domain shape:'.format(shape))

        out = out.reshape(shape)
        nt = shape[-1]

        fig = plt.figure(dpi=600, figsize=(10, 10))
        axs = [fig.add_subplot(3, 4, i + 1) for i in range(nt)]
        X, Y = self.domain.X, self.domain.Y

        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
            Y = Y.detach().cpu().numpy()

        for i in range(nt):
            value = out[:, :, i]
            gca = axs[i].pcolormesh(X[:, :, 0], Y[:, :, 0], out[:, :, i],
                                    shading='auto', cmap=plt.cm.jet)
            if add_point is not None:
                idx = np.where(np.all((i < add_point[:, -1],  add_point[:, -1] <= i+1), axis=0))[0]
                axs[i].plot(add_point[idx, 0], add_point[idx, 1],  'kx', markersize=3,
                            label='add {} points'.format(len(idx)), alpha=1.0)

            axs[i].set_aspect('equal')
            axs[i].set_xlabel('X')
            axs[i].set_ylabel('Y')
            axs[i].set_xlim(self.domain.x_min, self.domain.x_max)
            axs[i].set_ylim(self.domain.y_min, self.domain.y_max)
            axs[i].set_title('residual{}'.format(i))

            cbar = fig.colorbar(gca, ax=axs[i], orientation='vertical', extend='both',
                                ticks=np.linspace(value.min(), value.max(), 5, endpoint=True), format='%.2f',
                                label='Residual Values')
            # 设置 colorbar 的刻度标签大小
            cbar.ax.tick_params(labelsize=2)


        # x, y = self.domain.bound_point[:, 0], self.domain.bound_point[:, 1]
        #
        # if isinstance(x, torch.Tensor):
        #     x = x.detach().cpu().numpy()
        #     y = y.detach().cpu().numpy()
        #
        # for i in range(nt):
        #     axs[i].set_xlabel('X')
        #     axs[i].set_ylabel('Y')
        #
        #     # rand_ind = [np.random.randint(0, len(self.bound_point)) for _ in range(20)]
        #     # use
        #
        #     im = axs[i].imshow(out[:, :, i], cmap='viridis', extent=[x.min(), x.max(), y.min(), y.max()],
        #                        origin='lower')
        #     axs[i].set_xlim(x.min(), x.max())
        #     axs[i].set_ylim(y.min(), y.max())
        #     # axs[i].plot(x, y, 'kx', markersize=5, clip_on=False, alpha=1.0)
        #     axs[i].set_title('residual_t{}'.format(i))
        #     if i == nt-1:
        #         cbar = fig.colorbar(im, ax=axs[i], orientation='vertical', extend='both',
        #                             ticks=np.linspace(0, res_max, 5), format='%.2f', label='Residual Values')
        #         # 设置 colorbar 的刻度标签大小
        #         cbar.ax.tick_params(labelsize=10)
        # plt.show()
        return fig


# class GMM:
def plot_10_figs(input_tensor, nx, ny, file_path, show=False):
    n = len(input_tensor)
    for t in range(n):
        if t % 10 == 0:
            fig = plt.figure(dpi=600, figsize=(9, 9))

        axs = fig.add_subplot(4, 3, (t % 10) + 1)
        if isinstance(input_tensor, torch.Tensor):
            out = input_tensor[t].detach().cpu().numpy().reshape(nx, ny)
        else:
            out = input_tensor[t].reshape(nx, ny)
        gca = axs.imshow(out, origin='lower', cmap='viridis')
        axs.set_xlabel('X', fontsize=5)
        axs.set_ylabel('Y', fontsize=5)
        axs.set_title('press_t_{}'.format(t + 1), fontsize=5)
        axs.tick_params(axis='both', labelsize=5)
        cbar = fig.colorbar(gca, ax=axs, orientation='vertical', extend='both',
                            ticks=np.linspace(out.min(), out.max(), 5, endpoint=True),
                            format='%.2f')  # ,label='Press Values'
        # 设置 colorbar 的刻度标签大小
        cbar.ax.tick_params(labelsize=2)

        if (t + 1) % 10 == 0:
            fig.suptitle(file_path[10:-4])
            plt.savefig(file_path)
            if show:
                plt.show()
        
    return fig


def plot_press(input_tensor, nx, ny, fig_name):
    fig = plt.figure(dpi=300)
    axs = fig.add_subplot(1,1,1)
    if isinstance(input_tensor, torch.Tensor):
            out = input_tensor.detach().cpu().numpy().reshape(nx, ny)
    else:
        out = input_tensor.reshape(nx, ny)
    gca = axs.imshow(out, origin='lower', cmap='viridis')
    axs.set_xlabel('X', fontsize=5)
    axs.set_ylabel('Y', fontsize=5)
    axs.set_title(fig_name, fontsize=5)
    axs.tick_params(axis='both', labelsize=5)
    cbar = fig.colorbar(gca, ax=axs, orientation='vertical', extend='both',
                        ticks=np.linspace(out.min(), out.max(), 5, endpoint=True),
                        format='%.2f')  # ,label='Press Values'
    # 设置 colorbar 的刻度标签大小
    cbar.ax.tick_params(labelsize=2)
    plt.savefig('./figure/'+fig_name)
