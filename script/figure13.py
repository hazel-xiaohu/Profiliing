import math
import numpy as np
import matplotlib.pyplot as plt
# This restores the same behavior as before.
import pandas as pd
import sys

label_font_size = 120
dot_size = 70
reduction_number = 5
SSSM_reduction_number = 20
legend_size = 110
mark_scale_size = 6
grid_alpha = 0.8
text_font_size = 80
rotation = 20
line_width = 15
colors = ["lightcoral", "cornflowerblue", "#f8cb7f"]


# no_select = list(map(float, df.no_select.values.tolist()))
pangulu_file_name_01=""
superlu_file_name_01=""
pangulu_file_name_02=""
superlu_file_name_02=""

def speedup(m, n):
    k = [1] * len(n)
    for i in range(len(m)):
        k[i] = m[i]/n[i]
    return k


def draw_Pangu(plt):

    df = pd.read_csv(pangulu_file_name_01)
    df2 = pd.read_csv(superlu_file_name_01)
    matrix = list(map(str, df.matrix.values.tolist()))
    SuperLU_DIST = list(
        map(float, df2.superlu_commu_syn.values.tolist()))
    PanguLU_origin = list(
        map(float, df.pangu_commu_syn_free.values.tolist()))
    # PanguLU_optimise = list(
    #     map(float, df.pangu_commu_syn_free.values.tolist()))

    x = np.arange(len(matrix))
    width = 0.25

    plt.bar(x - width/2, SuperLU_DIST, width, color=colors[0],
            edgecolor='black', linewidth=line_width, label='SuperLU_DIST', zorder=102)
    plt.bar(x + width/2, PanguLU_origin, width, color=colors[1],
            edgecolor='black', linewidth=line_width, label='PanguLU_Original', zorder=102)
    # plt.bar(x + width/2, PanguLU_optimise, width, color=colors[2],
    #         edgecolor='black', linewidth=line_width, label='PanguLU', zorder=102)

    # select_nosynchonrize_1 = speedup(no_select, PanguLU_Kernel_Selection)
    # select_1 = speedup(no_select, PanguLU_Origin)
    # no_select_1 = speedup(no_select, no_select)

    PanguLU_origin_1 = PanguLU_origin
    SuperLU_DIST_1 = SuperLU_DIST
    # PanguLU_optimise_1 = PanguLU_optimise

    # print(no_select)
    # print(PanguLU_Kernel_Selection)

    for a, b, c in zip(x, SuperLU_DIST, SuperLU_DIST_1): 
        plt.text(a-width/2, b+0.3, '%.1f' % c, ha='center',
                 va='bottom', rotation=0, fontsize=text_font_size, zorder=102)

    for a, b, c in zip(x, PanguLU_origin, PanguLU_origin_1): 
        plt.text(a + width/2, b+0.3, '%.1f' % c, ha='center',
                 va='bottom', rotation=0, fontsize=text_font_size, zorder=102)

    # for a, b, c in zip(x, PanguLU_optimise, PanguLU_optimise_1): 
    #     plt.text(a+width/2, b+0.3, '%.1f' % c, ha='center',
    #              va='bottom', rotation=0, fontsize=text_font_size, zorder=102)

    plt.grid(zorder=0)
    plt.tick_params(axis='both', labelsize=label_font_size)
    plt.set_ylabel('Sync Time (s)', fontsize=label_font_size+30)
    # plt.set_xlabel('Synchronization time comparison between PanguLU_Origin and SuperLU_DIST',
    #                fontsize=label_font_size+70)

    max_data=max(max(SuperLU_DIST),max(PanguLU_origin))
    plt.set_ylim(0, max_data+5)

    plt.set_xticks(x)
    plt.set_xticklabels(matrix, rotation=rotation,
                        fontsize=label_font_size*1.35)
    plt.legend(loc='upper left',
               borderaxespad=0, fontsize=legend_size*1.2, ncol=1)


def draw_Pangu2(plt):
    df = pd.read_csv(pangulu_file_name_02)
    df2 = pd.read_csv(superlu_file_name_02)
    matrix = list(map(str, df.matrix.values.tolist()))
    SuperLU_DIST = list(
        map(float, df2.superlu_commu_syn.values.tolist()))
    PanguLU_origin = list(
        map(float, df.pangu_commu_syn_free.values.tolist()))
    # PanguLU_optimise = list(
    #     map(float, df.pangu_commu_syn_free.values.tolist()))

    x = np.arange(len(matrix))
    width = 0.25

    plt.bar(x - width/2, SuperLU_DIST, width, color=colors[0],
            edgecolor='black', linewidth=line_width, label='SuperLU_DIST', zorder=102)
    plt.bar(x + width/2, PanguLU_origin, width, color=colors[1],
            edgecolor='black', linewidth=line_width, label='PanguLU', zorder=102)
    # plt.bar(x + width/2, PanguLU_optimise, width, color=colors[2],
    #         edgecolor='black', linewidth=line_width, label='PanguLU', zorder=102)

    # select_nosynchonrize_1 = speedup(no_select, PanguLU_Kernel_Selection)
    # select_1 = speedup(no_select, PanguLU_Origin)
    # no_select_1 = speedup(no_select, no_select)

    PanguLU_origin_1 = PanguLU_origin
    SuperLU_DIST_1 = SuperLU_DIST
    # PanguLU_optimise_1 = PanguLU_optimise

    # print(no_select)
    # print(PanguLU_Kernel_Selection)

    for a, b, c in zip(x, SuperLU_DIST, SuperLU_DIST_1): 
        plt.text(a - width/2, b + 0.3, '%.1f' % c, ha='center',
                 va='bottom', rotation=0, fontsize=text_font_size, zorder=102)

    for a, b, c in zip(x, PanguLU_origin, PanguLU_origin_1): 
        plt.text(a + width/2, b + 0.3, '%.1f' % c, ha='center',
                 va='bottom', rotation=0, fontsize=text_font_size, zorder=102)

    # for a, b, c in zip(x, PanguLU_optimise, PanguLU_optimise_1): 
    #     plt.text(a + width/2, b + 0.3, '%.1f' % c, ha='center',
    #              va='bottom', rotation=0, fontsize=text_font_size, zorder=102)

    plt.grid(zorder=0)
    plt.tick_params(axis='both', labelsize=label_font_size)
    plt.set_ylabel('Sync Time (s)', fontsize=label_font_size + 30)
    # plt.set_xlabel('Synchronization time comparison between PanguLU_Origin and SuperLU_DIST',
    #                fontsize=label_font_size+70)
    max_data=max(max(SuperLU_DIST),max(PanguLU_origin))
    plt.set_ylim(0, max_data+5)
    # plt.set_ylim(0, 30)

    plt.set_xticks(x)
    plt.set_xticklabels(matrix, rotation=rotation,
                        fontsize=label_font_size * 1.35)
    # plt.legend(bbox_to_anchor=(0.87, 0.995), loc='upper left',
    #            borderaxespad=0, fontsize=legend_size * 1.2, ncol=1)


def main():
    fig, axs = plt.subplots(2, 1, dpi=7, figsize=(110, 55))
    plt.subplots_adjust(hspace=0.5, wspace=0.1)
    for ax in axs.flatten():
        for pos in ['top', 'bottom', 'left', 'right']:
            ax.spines[pos].set_linewidth(10)
    draw_Pangu(axs[0])
    draw_Pangu2(axs[1])
    # draw_Pangu_superlu(axs[1])

    plt.savefig("../figure/figure13.pdf",
                bbox_inches='tight', dpi=300)

    #plt.show()


if __name__ == '__main__':
    pangulu_file_name_01 = sys.argv[1]
    superlu_file_name_01 = sys.argv[2]
    pangulu_file_name_02 = sys.argv[3]
    superlu_file_name_02 = sys.argv[4]
    main()
