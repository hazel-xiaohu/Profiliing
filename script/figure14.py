import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

label_font_size = 88
dot_size = 70
reduction_number = 5
SSSM_reduction_number = 20
legend_size = 120
mark_scale_size = 6
grid_alpha = 0.8
text_font_size = 80
rotation = 16

df1 = pd.read_csv(sys.argv[1])
matrix1 = list(map(str, df1.matrix.values.tolist()))
select1 = list(map(float, df1.select.values.tolist()))
select_nosynchonrize1 = list(
    map(float, df1.select_nosynchonrize.values.tolist()))
no_select1 = list(map(float, df1.no_select.values.tolist()))

df2 = pd.read_csv(sys.argv[2])
matrix2 = list(map(str, df2.matrix.values.tolist()))
select2 = list(map(float, df2.select.values.tolist()))
select_nosynchonrize2 = list(
    map(float, df2.select_nosynchonrize.values.tolist()))
no_select2 = list(map(float, df2.no_select.values.tolist()))


def speedup(m, n):
    k = [1] * len(n)
    for i in range(len(m)):
        k[i] = round(float(m[i]/n[i]), 2)
    return k


def draw_Pangu(plt, num):
    if (num == 0):
        x = np.arange(len(matrix1))
        width = 0.25

        no_select = speedup(no_select1, no_select1)
        select = speedup(no_select1, select1)
        select_nosynchonrize = speedup(no_select1, select_nosynchonrize1)

        plt.bar(x - width, no_select, width, color='mistyrose',
                edgecolor='black', linewidth=2.5, label='Original', zorder=102)
        plt.bar(x, select, width, color='salmon', edgecolor='black',
                linewidth=2.5, label='Kernel selection', zorder=102)
        plt.bar(x + width, select_nosynchonrize, width, color='orangered',
                edgecolor='black', linewidth=2.5, label='Kernel selection & Synchronisation free', zorder=102)
        for a, b, c in zip(x, no_select, no_select): 
            plt.text(a - width, b + 0.05, '%.1f' % c, ha='center',
                     va='bottom', fontsize=text_font_size*2.5, zorder=102)

        for a, b, c in zip(x, select, select):
            if (b >= 12.8):
                plt.text(a - 0.02, b + 0.05, '%.1f' % c, ha='center', va='bottom',
                         fontsize=text_font_size * 2.5, zorder=102)
            else:
                plt.text(a, b + 0.05, '%.1f' % c, ha='center', va='bottom',
                         fontsize=text_font_size * 2.5, zorder=102)
        for a, b, c in zip(x, select_nosynchonrize, select_nosynchonrize):  
            plt.text(a + width, b + 0.05, '%.1f' % c, ha='center',
                     va='bottom', fontsize=text_font_size*2.5, zorder=102)

        plt.grid(zorder=0)
        plt.tick_params(axis='both', labelsize=label_font_size*3)
        plt.set_ylabel('Speedup', fontsize=label_font_size*3)
        plt.set_ylim(0, 10)
        plt.set_xticks(x)
        plt.set_xticklabels(matrix1, rotation=rotation,
                            fontsize=label_font_size*3)

        plt.legend(bbox_to_anchor=(0.548, 0.98), loc='upper right',
                   borderaxespad=0, fontsize=legend_size*2.4, ncol=1)
    else:
        x = np.arange(len(matrix2))
        width = 0.25

        no_select = speedup(no_select2, no_select2)
        select = speedup(no_select2, select2)
        select_nosynchonrize = speedup(no_select2, select_nosynchonrize2)

        plt.bar(x - width, no_select, width, color='mistyrose',
                edgecolor='black', linewidth=2.5, label='Original', zorder=102)
        plt.bar(x, select, width, color='salmon', edgecolor='black',
                linewidth=2.5, label='Kernel selection', zorder=102)
        plt.bar(x + width, select_nosynchonrize, width, color='orangered',
                edgecolor='black', linewidth=2.5, label='Kernel selection & Synchronisation free', zorder=102)
        for a, b, c in zip(x, no_select, no_select): 
            plt.text(a - width, b + 0.05, '%.1f' % c, ha='center',
                     va='bottom', fontsize=text_font_size*2.5, zorder=102)
        for a, b, c in zip(x, select, select):
            plt.text(a, b + 0.05, '%.1f' % c, ha='center', va='bottom',
                        fontsize=text_font_size * 2.5, zorder=102)

        for a, b, c in zip(x, select_nosynchonrize, select_nosynchonrize): 
            plt.text(a + width, b + 0.05, '%.1f' % c, ha='center',
                     va='bottom', fontsize=text_font_size*2.5, zorder=102)

        plt.grid(zorder=0)
        plt.tick_params(axis='both', labelsize=label_font_size*3)
        plt.set_ylabel('Speedup', fontsize=label_font_size*3)
        plt.set_xlabel('Performance comparison between three versions of PanguLU', loc="center",
                       fontsize=label_font_size*4)
        plt.set_ylim(0, 5.2)
        plt.set_xticks(x)
        plt.set_xticklabels(matrix2, rotation=rotation,
                            fontsize=label_font_size*3)


def main():
    fig, axes = plt.subplots(nrows=2, ncols=1, dpi=6, figsize=(220, 145))
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    for ax in axes.flatten():
        for pos in ['top', 'bottom', 'left', 'right']:
            ax.spines[pos].set_linewidth(10)
    draw_Pangu(axes[0], 0)
    draw_Pangu(axes[1], 1)

    plt.savefig("../figure/figure14.pdf",
                bbox_inches='tight', dpi=300)

    #plt.show()


if __name__ == '__main__':
    main()
