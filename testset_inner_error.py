import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.PDE_module import plot_10_figs, check_devices
from src.Mesh import *
import seaborn as sns
import argparse


def read_file(filepath):
    train_data = np.load(filepath)
    train_permeability, test_permeability = train_data['train_permeability'], train_data['test_permeability']
    # train_label_press, test_label_press = train_data['train_label_press'], train_data['test_label_press']
    train_label_press, test_label_press = train_data['train_press'], train_data['test_press']
    return train_permeability, test_permeability, train_label_press, test_label_press


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_name', default='./dataset/sg_4well.npz')
parser.add_argument('-wn', '--well_num', default=4, type=int)
parser.add_argument('-n', '--nt', default=150, type=int)
args = parser.parse_args()


bhp_constant = 20 * 1e6
pwf = bhp_constant

train_permeability, test_permeability, train_label_press, test_label_press = read_file(args.file_name)
re_mean_error = []
re_well_error = []
qwt = np.zeros((args.nt, len(test_permeability), args.well_num))
if args.well_num == 1:
    well_mark = [0]
else:
    well_mark = [0, 19, 380, 399]

shape = test_label_press.shape
print('label press length: {}, test {} time step data'.format(shape[1], args.nt))
print('data num: {}'.format(shape[0] * shape[1]))
# print('test {} time step data'.format(args.nt))
p_init = 30 * 1e6
for t in range(args.nt):
    label_press_t = test_label_press[:, t, :]
    press_t_mean = label_press_t.mean(axis=0)
    relative_error = np.abs((label_press_t - press_t_mean) / (p_init - press_t_mean))
    
    for re in relative_error:
        re_mean = re.mean()
        re_well = re[well_mark]
        if re_mean < 1e-10:
            re_mean_error.append(1e-10)
        else:
            re_mean_error.append(re_mean)

        if re_well.mean() < 1e-10:
            re_well_error.append(1e-10)
        else:
            re_well_error.append(re_well.mean())

    qw = (label_press_t[:, well_mark] - pwf)  # shape: [420, 4]; mean:
    qwt[t] = qw


re_mean_error = np.array(re_mean_error)
re_well_error = np.array(re_well_error)
log_mean_error = np.log10(re_mean_error)
log_well_error = np.log10(re_well_error)
print('mean relative error: {}'.format(re_mean_error.mean()))
print('well relative error: {}'.format(re_well_error.mean()))

data = {'log mean relative error': log_mean_error, 'log well relative error': log_well_error}
df = pd.DataFrame(data)
sns.set(style="ticks", palette="muted")
sns.displot(data=df, x='log mean relative error', kde=True, bins=100)
plt.title('log mean data inner re_error')
plt.savefig('./figure/{}well_data_mean_error.png'.format(args.well_num), dpi=400, bbox_inches='tight')
plt.show()

sns.displot(data=df, x='log well relative error', kde=True, bins=100)
plt.title('log well data inner re_error')
plt.savefig('./figure/{}well_data_well_error.png'.format(args.well_num), dpi=400, bbox_inches='tight')
plt.show()

# write data into textfile
df.to_csv('./dataset/{}well_nt{}_data_re.txt'.format(args.well_num, args.nt), sep='\t', index=False, header=True)
mean_qwt = qwt.mean(axis=1)
mean_qwt_sum = mean_qwt.cumsum(axis=0)
FVM_qwt_mean = pd.DataFrame({'FVM_{}'.format(i): mean_qwt_sum[:, i] for i in range(4)})
sns.relplot(data=mean_qwt_sum, kind='line')
plt.xlabel('time step')
plt.ylabel('accumulative qwt')
plt.legend(title='well number')
plt.savefig('./FVM_qwt_mean_sum.png')
plt.show()
FVM_qwt_mean.to_csv('./dataset/{}well_nt{}_qwt_sum.txt'.format(args.well_num, args.nt), sep='\t', index=False, header=True)