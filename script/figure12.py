import matplotlib.pyplot as plt
import numpy as np
import csv
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


styles = ['c:s', 'y:8', 'r:^', 'r:v', 'g:D', 'm:X', 'p', ':>', '*']
style_index = 8
pangu_index = 6


color_index = 0

color_2 = '#0780CF'
color_1 = '#DA1F18'
point_s = 15



df = pd.read_csv(sys.argv[1])
data = df.values
x = [1, 2, 3, 4, 5, 6, 7, 8]

df_superlu = pd.read_csv(sys.argv[2])
data_superlu = df_superlu.values
x = [1, 2, 3, 4, 5, 6, 7, 8]

sizeofr = 8
sizeofc = 9
data_superlu_gpu = [0.0 for i in range(sizeofr)]
data_pangu = [0.0 for i in range(sizeofr)]
divide1 = [1.03, 1.2, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03, 1.03]
divide2 = [1.01, 0.96, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01, 1.01]

stand1 = [1, 2, 4, 8, 16, 32, 64, 128]



def draw_picture(ax, x, xt, data_pangu, data_superlu_gpu, index, divide1, divide2):
    big = 0.0
    for num in range(0, 8):
        if (big < data_pangu[num]):
            big = data_pangu[num]
    ax.ylim([0, big*1.1])
    ax.grid(zorder=0)
    ax.plot(x, data_pangu, marker=styles[pangu_index], markersize=point_s, linestyle='solid',  color=color_1,
            label="PanguLU", linewidth=4)
    
    ax.plot(x, data_superlu_gpu, marker=styles[style_index], markersize=point_s+2,
            color=color_2, linestyle='solid',  label="SuperLU-DIST", linewidth=4)
    for a, b, c in zip(x, data_pangu, data_pangu):

        plt.text(a-0.08, b*divide1[a], '%.1f' % c, ha='center',
                 va='bottom', rotation=0, fontsize=25, zorder=152)

    for a, b, c in zip(x, data_superlu_gpu, data_superlu_gpu):  # 控制标签位置
        plt.text(a-0.08, b*divide2[a], '%.1f' % c, ha='center',
                 va='bottom', rotation=0, fontsize=25, zorder=152)
    ax.xticks(x, xt, fontsize=30)
    ax.yticks(fontsize=30)
    ax.xlabel(data[index][0], fontsize=60)
    ax.xlim([0.3, 8.7])
    return big

def get_data(index):
    for num in range(sizeofr):
        data_pangu[num] = (float)(data[index][num+1])

def get_data_superlu(index):
    for num in range(sizeofr):
        data_superlu_gpu[num] = (float)(data_superlu[index][num+1])



    

length = 40

fig = plt.figure(figsize=(length, length/1.35))

write_size = 70

xt = ['1', '2', '4', '8', '16', '32', '64', '128']


def draw():
    plt.subplots_adjust(left=None, bottom=None,
                        right=None, top=None, hspace=0.3)
    # plt.ylabel("GFlops",fontsize=30)

    plt.subplot(4, 4, 1)
    index = 0
    get_data(index=index)
    get_data_superlu(index=index)
    big=draw_picture(ax=plt, x=x, xt=xt, data_pangu=data_pangu,
                 data_superlu_gpu=data_superlu_gpu, index=index, divide1=divide1, divide2=divide2)
    plt.ylim([0, big*1.4])
    # plt.ylabel("GFlops",fontsize=write_size)
    plt.text(-1.5, 25, 'GFlops', ha='center',
             va='bottom', rotation=90, fontsize=60, zorder=152)

    plt.subplot(4, 4, 2)
    index = 1
    get_data(index=index)
    get_data_superlu(index=index)
    big=draw_picture(ax=plt, x=x, xt=xt, data_pangu=data_pangu,
                 data_superlu_gpu=data_superlu_gpu, index=index, divide1=divide1, divide2=divide2)
    plt.ylim([0, big*1.4])
    # plt.ylabel("GFlops",fontsize=write_size)
    plt.legend(bbox_to_anchor=(0.65, 0.98), loc=1,
               borderaxespad=0, fontsize=37, ncol=1)

    plt.subplot(4, 4, 3)
    index = 2
    get_data(index=index)
    get_data_superlu(index=index)
    big=draw_picture(ax=plt, x=x, xt=xt, data_pangu=data_pangu,
                 data_superlu_gpu=data_superlu_gpu, index=index, divide1=divide1, divide2=divide2)
    plt.ylim([0, big*1.4])

    plt.subplot(4, 4, 4)
    index = 3
    get_data(index=index)
    get_data_superlu(index=index)
    big=draw_picture(ax=plt, x=x, xt=xt, data_pangu=data_pangu,
                 data_superlu_gpu=data_superlu_gpu, index=index, divide1=divide1, divide2=divide2)
    plt.ylim([0, big*1.4])

    plt.subplot(4, 4, 5)
    index = 4
    get_data(index=index)
    get_data_superlu(index=index)
    big=draw_picture(ax=plt, x=x, xt=xt, data_pangu=data_pangu,
                 data_superlu_gpu=data_superlu_gpu, index=index, divide1=divide1, divide2=divide2)
    plt.ylim([0, big*1.4])
    # plt.ylabel("GFlops",fontsize=write_size)
    plt.text(-1.5, 75, 'GFlops', ha='center',
             va='bottom', rotation=90, fontsize=60, zorder=152)

    plt.subplot(4, 4, 6)
    index = 5
    get_data(index=index)
    get_data_superlu(index=index)
    big=draw_picture(ax=plt, x=x, xt=xt, data_pangu=data_pangu,
                 data_superlu_gpu=data_superlu_gpu, index=index, divide1=divide1, divide2=divide2)
    plt.ylim([0, big*1.4])

    plt.subplot(4, 4, 7)
    index = 6
    get_data(index=index)
    get_data_superlu(index=index)
    big=draw_picture(ax=plt, x=x, xt=xt, data_pangu=data_pangu,
                 data_superlu_gpu=data_superlu_gpu, index=index, divide1=divide1, divide2=divide2)
    plt.ylim([0, big*1.4])
    # plt.ylabel("GFlops",fontsize=write_size)

    plt.subplot(4, 4, 8)
    index = 7
    get_data(index=index)
    get_data_superlu(index=index)
    big=draw_picture(ax=plt, x=x, xt=xt, data_pangu=data_pangu,
                 data_superlu_gpu=data_superlu_gpu, index=index, divide1=divide1, divide2=divide2)
    plt.ylim([0, 35])

    plt.subplot(4, 4, 9)
    index = 8
    get_data(index=index)
    get_data_superlu(index=index)
    big=draw_picture(ax=plt, x=x, xt=xt, data_pangu=data_pangu,
                 data_superlu_gpu=data_superlu_gpu, index=index, divide1=divide1, divide2=divide2)
    plt.ylim([0, big*1.4])
    # plt.ylabel("GFlops",fontsize=write_size)
    plt.text(-1.5, 750, 'GFlops', ha='center',
             va='bottom', rotation=90, fontsize=60, zorder=152)

    plt.subplot(4, 4, 10)
    index = 9
    get_data(index=index)
    get_data_superlu(index=index)
    big=draw_picture(ax=plt, x=x, xt=xt, data_pangu=data_pangu,
                 data_superlu_gpu=data_superlu_gpu, index=index, divide1=divide1, divide2=divide2)
    plt.ylim([0, big*1.4])
    # plt.ylabel("GFlops",fontsize=write_size)

    plt.subplot(4, 4, 11)
    index = 10
    get_data(index=index)
    get_data_superlu(index=index)
    big=draw_picture(ax=plt, x=x, xt=xt, data_pangu=data_pangu,
                 data_superlu_gpu=data_superlu_gpu, index=index, divide1=divide1, divide2=divide2)
    plt.ylim([0, 70])

    plt.subplot(4, 4, 12)
    index = 11
    get_data(index=index)
    get_data_superlu(index=index)
    big=draw_picture(ax=plt, x=x, xt=xt, data_pangu=data_pangu,
                 data_superlu_gpu=data_superlu_gpu, index=index, divide1=divide1, divide2=divide2)
    plt.ylim([0, big*1.4])

    plt.subplot(4, 4, 13)
    index = 12
    get_data(index=index)
    get_data_superlu(index=index)
    big=draw_picture(ax=plt, x=x, xt=xt, data_pangu=data_pangu,
                 data_superlu_gpu=data_superlu_gpu, index=index, divide1=divide1, divide2=divide2)
    plt.ylim([0, big*1.4])
    # plt.ylabel("GFlops",fontsize=write_size)
    plt.text(-1.5, 300, 'GFlops', ha='center',
             va='bottom', rotation=90, fontsize=60, zorder=152)

    plt.subplot(4, 4, 14)
    index = 13
    get_data(index=index)
    get_data_superlu(index=index)
    big=draw_picture(ax=plt, x=x, xt=xt, data_pangu=data_pangu,
                 data_superlu_gpu=data_superlu_gpu, index=index, divide1=divide1, divide2=divide2)
    plt.ylim([0, big*1.4])

    plt.subplot(4, 4, 15)
    index = 14
    get_data(index=index)
    get_data_superlu(index=index)
    big=draw_picture(ax=plt, x=x, xt=xt, data_pangu=data_pangu,
                 data_superlu_gpu=data_superlu_gpu, index=index, divide1=divide1, divide2=divide2)
    plt.ylim([0, big*1.4])

    plt.subplot(4, 4, 16)
    index = 15
    get_data(index=index)
    get_data_superlu(index=index)
    big=draw_picture(ax=plt, x=x, xt=xt, data_pangu=data_pangu,
                 data_superlu_gpu=data_superlu_gpu, index=index, divide1=divide1, divide2=divide2)
    plt.ylim([0, big*1.4])

    
    plt.tight_layout()

    plt.savefig("../figure/figure12.pdf",
                bbox_inches='tight', dpi=300)

if __name__ == '__main__':

    draw()
