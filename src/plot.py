import numpy as np
import torch
import matplotlib.pyplot as plt


def plot_10_figs(input_tensor, nx, ny, file_path, show=False, id_num=None):
    n = len(input_tensor)
    fig = None  # 初始化fig变量
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
        if id_num is None:
            axs.set_title('press_t_{}'.format(t + 1), fontsize=5)
        else:
            axs.set_title('id_{}press_t_{}'.format(id_num[t], t + 1), fontsize=5)
        axs.tick_params(axis='both', labelsize=5)
        cbar = fig.colorbar(gca, ax=axs, orientation='vertical', extend='both',
                            ticks=np.linspace(out.min(), out.max(), 5, endpoint=True),
                            format='%.2f')  # ,label='Press Values'
        # 设置 colorbar 的刻度标签大小
        cbar.ax.tick_params(labelsize=2)

        if (t + 1) % 10 == 0:
            fig.suptitle(file_path[9:-4])
            plt.savefig(file_path)
            if show:
                plt.show()
    # return fig


def show_filed(input_tensor, nx, ny, fig_name, fig_path='./figure/', show=False):
    fig = plt.figure(dpi=300)
    axs = fig.add_subplot(1 ,1 ,1)
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
    plt.savefig(fig_path + fig_name)
    if show:
        plt.show()
