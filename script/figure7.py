import math
import sys
import matplotlib.pyplot as plt
# This restores the same behavior as before.
import pandas as pd

label_font_size = 200
dot_size = 200
reduction_number = 6
SSSM_reduction_number = 5
GETRF_reduction_number = 2
legend_size = 160
mark_scale_size = 6
grid_alpha = 0.8
# cnt = 0

scatter_color = ['#e3852b', '#99cc00', '#e30039', '#800080', '#8085e9']

# C_1 C_2 G_1 G_2 G_3
GESSM_FILE_PATH =  sys.argv[1]
TSTRF_FILE_PATH =  sys.argv[2]
SSSSM_FILE_PATH =  sys.argv[3]
GETRF_FILE_PATH =  sys.argv[4]


def log_10(m):  # log
    data = m
    for i in range(0, len(m)):
        data[i] = m[i]
        if (data[i] <= 0):
            data[i] = 0
        else:
            data[i] = math.log10(data[i])
    return data


def get_max_num(m):
    data = -1
    for i in range(0, len(m)):
        if (data < m[i]):
            data = m[i]
    return data


def div_1000(m):
    data = m
    for i in range(0, len(m)):
        data[i] = m[i] / 1000.0
    return data


def change_getrf_data(m):
    cnt = 0
    data = m
    for i in range(0, len(m)):
        if (m[i] > 4.5):
            cnt += 1
            data[i] = m[i]-0.5

    return data


def array_reduction(m, number):
    n = int(len(m) / number)
    data = [float(0.0) for _ in range(n)]
    for i in range(0, n):
        data[i] = m[i * number]
    return data


def draw_TSTRF(plt2):
    df2 = pd.read_csv(TSTRF_FILE_PATH)
    nnz_B = list(map(float, df2.nnz_B.values.tolist()))
    cpu_1 = list(map(float, df2.cpu_1.values.tolist()))
    cpu_2 = list(map(float, df2.cpu_2.values.tolist()))
    cpu_3 = list(map(float, df2.cpu_3.values.tolist()))
    cpu_4 = list(map(float, df2.cpu_4.values.tolist()))
    cpu_5 = list(map(float, df2.cpu_5.values.tolist()))
    cpu_6 = list(map(float, df2.cpu_6.values.tolist()))
    gpu_7 = list(map(float, df2.gpu_7.values.tolist()))
    gpu_8 = list(map(float, df2.gpu_8.values.tolist()))
    gpu_9 = list(map(float, df2.gpu_9.values.tolist()))
    gpu_10 = list(map(float, df2.gpu_10.values.tolist()))
    gpu_11 = list(map(float, df2.gpu_11.values.tolist()))

    nnz_B = array_reduction(nnz_B, reduction_number)
    cpu_1 = array_reduction(cpu_1, reduction_number)
    cpu_2 = array_reduction(cpu_2, reduction_number)
    cpu_3 = array_reduction(cpu_3, reduction_number)
    cpu_4 = array_reduction(cpu_4, reduction_number)
    cpu_5 = array_reduction(cpu_5, reduction_number)
    cpu_6 = array_reduction(cpu_6, reduction_number)
    gpu_7 = array_reduction(gpu_7, reduction_number)
    gpu_8 = array_reduction(gpu_8, reduction_number)
    gpu_9 = array_reduction(gpu_9, reduction_number)
    gpu_10 = array_reduction(gpu_10, reduction_number)
    gpu_11 = array_reduction(gpu_11, reduction_number)

    cpu_1 = log_10(cpu_1)
    cpu_2 = log_10(cpu_2)
    cpu_3 = log_10(cpu_3)
    cpu_4 = log_10(cpu_4)
    cpu_5 = log_10(cpu_5)
    cpu_6 = log_10(cpu_6)
    gpu_7 = log_10(gpu_7)
    gpu_8 = log_10(gpu_8)
    gpu_9 = log_10(gpu_9)
    gpu_10 = log_10(gpu_10)
    gpu_11 = log_10(gpu_11)
    nnz_B = log_10(nnz_B)
    max_nnz_B = get_max_num(nnz_B)
    # plt2.scatter(x=nnz_B,
    #              y=cpu_1,
    #              s=dot_size,
    #              marker='o',
    #              color='red',
    #              alpha=1,
    #              label='C_V1'
    #              # fontsize = 30
    #              )
    # plt2.scatter(x=nnz_B,
    #              y=cpu_2,
    #              s=dot_size,
    #              marker='o',
    #              color='orange',
    #              alpha=1,
    #              label='C_V2',
    #              # fontsize = 30
    #              )
    # plt2.scatter(x=nnz_B,
    #              y=cpu_3,
    #              s=dot_size,
    #              marker='o',
    #              color='Magenta',
    #              alpha=1,
    #              label='C_V3',
    #              # fontsize = 30
    #              )
    plt2.scatter(x=nnz_B,
                 y=cpu_4,
                 s=dot_size,
                 marker='o',
                 color=scatter_color[0],
                 alpha=1,
                 label='C_V1',
                 # fontsize = 30
                 )
    # plt2.scatter(x=nnz_B,
    #              y=cpu_5,
    #              s=dot_size,
    #              marker='o',
    #              color='violet',
    #              alpha=1,
    #              label='C_V5',
    #              # fontsize = 30
    #              )
    plt2.scatter(x=nnz_B,
                 y=cpu_6,
                 s=dot_size,
                 marker='o',
                 color=scatter_color[1],
                 alpha=1,
                 label='C_V2',
                 # fontsize = 30
                 )
    plt2.scatter(x=nnz_B,
                 y=gpu_7,
                 s=dot_size,
                 marker='o',
                 color=scatter_color[2],
                 alpha=1,
                 label='G_V1',
                 # fontsize = 30
                 )
    plt2.scatter(x=nnz_B,
                 y=gpu_8,
                 s=dot_size,
                 marker='o',
                 color=scatter_color[3],
                 alpha=1,
                 label='G_V2',
                 # fontsize = 30
                 )
    # plt2.scatter(x=nnz_B,
    #              y=gpu_9,
    #              s=dot_size,
    #              marker='o',
    #              color='blue',
    #              alpha=1,
    #              label='G_V3',
    #              # fontsize = 30
    #              )
    plt2.scatter(x=nnz_B,
                 y=gpu_10,
                 s=dot_size,
                 marker='o',
                 color=scatter_color[4],
                 alpha=1,
                 label='G_V3',
                 # fontsize = 30
                 )
    # plt2.scatter(x=nnz_B,
    #              y=gpu_11,
    #              s=dot_size,
    #              marker='o',
    #              color='cyan',
    #              alpha=1,
    #              label='G_V5',
    #              # fontsize = 30
    #              )

    plt2.tick_params(axis='both', labelsize=label_font_size)
    # plt2.xlim(0, 6.5)
    # plt2.set_title("SpTRSM", fontsize=label_font_size)
    plt2.set_xlabel("nnz_B ($\\log_{10}$)"+"\n"+"(b) TSTRF",
                    fontsize=label_font_size)
    plt2.set_ylabel("Time ($\\log_{10}$(ms))", fontsize=label_font_size)
    plt2.grid(c='black', alpha=grid_alpha, linestyle='--')
    plt2.ticklabel_format(style='plain', axis='x')
    plt2.legend(framealpha=0, ncol=3, loc='upper left',
                markerscale=mark_scale_size, fontsize=legend_size)


def draw_GETRF(plt):
    df2 = pd.read_csv(GETRF_FILE_PATH)

    nnz = list(map(float, df2.nnz.values.tolist()))
    cpu_dense = list(map(float, df2.cpu_dense.values.tolist()))
    gpu_dense = list(map(float, df2.gpu_dense.values.tolist()))
    gpu_sparse = list(map(float, df2.gpu_sparse.values.tolist()))

    nnz = array_reduction(nnz, GETRF_reduction_number)
    cpu_dense = array_reduction(cpu_dense, GETRF_reduction_number)
    gpu_dense = array_reduction(gpu_dense, GETRF_reduction_number)
    gpu_sparse = array_reduction(gpu_sparse, GETRF_reduction_number)
    cpu_dense = log_10(cpu_dense)
    gpu_dense = log_10(gpu_dense)
    gpu_sparse = log_10(gpu_sparse)

    nnz = log_10(nnz)
    # print(cnt)

    # print(nnz)

    plt.scatter(x=nnz,
                y=cpu_dense,
                s=dot_size,
                marker='o',
                color=scatter_color[0],
                alpha=1,
                label='C_V1'
                # fontsize = 30
                )
    # plt.scatter(x=nnz,
    #             y=cpu_thmkl,
    #             s=dot_size,
    #             marker='o',
    #             color='steelblue',
    #             alpha=1 ,
    #             label='cpu_sparse (omp=4)',
    #             # fontsize = 30
    #             )
    plt.scatter(x=nnz,
                y=gpu_sparse,
                s=dot_size,
                marker='o',
                color=scatter_color[2],
                alpha=1,
                label='G_V1',
                # fontsize = 30
                )
    plt.scatter(x=nnz,
                y=gpu_dense,
                s=dot_size,
                marker='o',
                color=scatter_color[3],
                alpha=1,
                label='G_V2',
                # fontsize = 30
                )

    # plt.set_title("LU Factorization", fontsize=label_font_size)
    plt.set_ylim(-1.4, 4.1)
    plt.tick_params(axis='both', labelsize=label_font_size)
    plt.set_xlabel("nnz_A ($\\log_{10}$)"+"\n" +
                   "(a) GETRF", fontsize=label_font_size)
    plt.set_ylabel("Time ($\\log_{10}$(ms))", fontsize=label_font_size)
    plt.grid(c='black', alpha=grid_alpha, linestyle='--')
    plt.ticklabel_format(style='plain', axis='x')
    plt.legend(framealpha=0, loc='upper left',
               markerscale=mark_scale_size, fontsize=legend_size)


def draw_SSSSM(plt):
    df2 = pd.read_csv(SSSSM_FILE_PATH)
    calculation_sum = list(map(float, df2.calcalution_sum.values.tolist()))
    cpu_dense = list(map(float, df2.cpu_dense.values.tolist()))
    cpu_thmkl = list(map(float, df2.cpu_thmkl.values.tolist()))
    gpu_dense = list(map(float, df2.gpu_dense.values.tolist()))
    gpu_sparse = list(map(float, df2.gpu_sparse.values.tolist()))

    calculation_sum = array_reduction(calculation_sum, SSSM_reduction_number)
    cpu_dense = array_reduction(cpu_dense, SSSM_reduction_number)
    cpu_thmkl = array_reduction(cpu_thmkl, SSSM_reduction_number)
    gpu_dense = array_reduction(gpu_dense, SSSM_reduction_number)
    gpu_sparse = array_reduction(gpu_sparse, SSSM_reduction_number)

    cpu_dense = log_10(cpu_dense)
    cpu_thmkl = log_10(cpu_thmkl)
    gpu_dense = log_10(gpu_dense)
    gpu_sparse = log_10(gpu_sparse)
    calculation_sum = log_10(calculation_sum)

    max_calculation_sum = get_max_num(calculation_sum)
    plt.scatter(x=calculation_sum,
                y=cpu_dense,
                s=dot_size,
                marker='o',
                color=scatter_color[0],
                alpha=1,
                label='C_V1'
                # fontsize = 30
                )
    plt.scatter(x=calculation_sum,
                y=cpu_thmkl,
                s=dot_size,
                marker='o',
                color=scatter_color[1],
                alpha=1,
                label='C_V2',
                # fontsize = 30
                )

    plt.scatter(x=calculation_sum,
                y=gpu_sparse,
                s=dot_size,
                marker='o',
                color=scatter_color[2],
                alpha=1,
                label='G_V1',
                # fontsize = 30
                )
    plt.scatter(x=calculation_sum,
                y=gpu_dense,
                s=dot_size,
                marker='o',
                color=scatter_color[3],
                alpha=1,
                label='G_V2',
                # fontsize = 30
                )
    plt.tick_params(axis='both', labelsize=label_font_size)
    # plt.title('SSSSM kenel time', fontsize=50)
    # plt.set_title("Schur-complement", fontsize=label_font_size)
    plt.set_xlabel("FLOPs ($\\log_{10}$)"+"\n" +
                   "(d) SSSSM", fontsize=label_font_size)
    plt.set_ylabel("Time ($\\log_{10}$(ms))", fontsize=label_font_size)
    plt.grid(c='black', alpha=grid_alpha, linestyle='--')
    plt.ticklabel_format(style='plain', axis='x')
    plt.legend(framealpha=0, ncol=2, loc='upper left',
               markerscale=mark_scale_size, fontsize=legend_size)


def draw_GESSM(plt2):
    df2 = pd.read_csv(GESSM_FILE_PATH)

    nnz_B = list(map(float, df2.nnz_B.values.tolist()))
    cpu_1 = list(map(float, df2.cpu_1.values.tolist()))
    cpu_2 = list(map(float, df2.cpu_2.values.tolist()))
    cpu_3 = list(map(float, df2.cpu_3.values.tolist()))
    cpu_4 = list(map(float, df2.cpu_4.values.tolist()))
    cpu_5 = list(map(float, df2.cpu_5.values.tolist()))
    cpu_6 = list(map(float, df2.cpu_6.values.tolist()))
    gpu_7 = list(map(float, df2.gpu_7.values.tolist()))
    gpu_8 = list(map(float, df2.gpu_8.values.tolist()))
    gpu_9 = list(map(float, df2.gpu_9.values.tolist()))
    gpu_10 = list(map(float, df2.gpu_10.values.tolist()))
    gpu_11 = list(map(float, df2.gpu_11.values.tolist()))

    nnz_B = array_reduction(nnz_B, reduction_number)
    cpu_1 = array_reduction(cpu_1, reduction_number)
    cpu_2 = array_reduction(cpu_2, reduction_number)
    cpu_3 = array_reduction(cpu_3, reduction_number)
    cpu_4 = array_reduction(cpu_4, reduction_number)
    cpu_5 = array_reduction(cpu_5, reduction_number)
    cpu_6 = array_reduction(cpu_6, reduction_number)
    gpu_7 = array_reduction(gpu_7, reduction_number)
    gpu_8 = array_reduction(gpu_8, reduction_number)
    gpu_9 = array_reduction(gpu_9, reduction_number)
    gpu_10 = array_reduction(gpu_10, reduction_number)
    gpu_11 = array_reduction(gpu_11, reduction_number)

    cpu_1 = log_10(cpu_1)
    cpu_2 = log_10(cpu_2)
    cpu_3 = log_10(cpu_3)
    cpu_4 = log_10(cpu_4)
    cpu_5 = log_10(cpu_5)
    cpu_6 = log_10(cpu_6)
    gpu_7 = log_10(gpu_7)
    gpu_8 = log_10(gpu_8)
    gpu_9 = log_10(gpu_9)
    gpu_10 = log_10(gpu_10)
    gpu_11 = log_10(gpu_11)
    nnz_B = log_10(nnz_B)
    max_nnz_B = get_max_num(nnz_B)

    # plt2.scatter(x=nnz_B,
    #              y=cpu_1,
    #              s=dot_size,
    #              marker='o',
    #              color='red',
    #              alpha=1,
    #              label='C_V1'
    #              # fontsize = 30
    #              )
    # plt2.scatter(x=nnz_B,
    #              y=cpu_2,
    #              s=dot_size,
    #              marker='o',
    #              color='orange',
    #              alpha=1,
    #              label='C_V2',
    #              # fontsize = 30
    #              )
    # plt2.scatter(x=nnz_B,
    #              y=cpu_3,
    #              s=dot_size,
    #              marker='o',
    #              color='Magenta',
    #              alpha=1,
    #              label='C_V3',
    #              # fontsize = 30
    #              )
    plt2.scatter(x=nnz_B,
                 y=cpu_4,
                 s=dot_size,
                 marker='o',
                 color=scatter_color[0],
                 alpha=1,
                 label='C_V1',
                 # fontsize = 30
                 )
    # plt2.scatter(x=nnz_B,
    #              y=cpu_5,
    #              s=dot_size,
    #              marker='o',
    #              color='violet',
    #              alpha=1,
    #              label='C_V5',
    #              # fontsize = 30
    #              )
    plt2.scatter(x=nnz_B,
                 y=cpu_6,
                 s=dot_size,
                 marker='o',
                 color=scatter_color[1],
                 alpha=1,
                 label='C_V2',
                 # fontsize = 30
                 )
    plt2.scatter(x=nnz_B,
                 y=gpu_7,
                 s=dot_size,
                 marker='o',
                 color=scatter_color[2],
                 alpha=1,
                 label='G_V1',
                 # fontsize = 30
                 )
    plt2.scatter(x=nnz_B,
                 y=gpu_8,
                 s=dot_size,
                 marker='o',
                 color=scatter_color[3],
                 alpha=1,
                 label='G_V2',
                 # fontsize = 30
                 )
    # plt2.scatter(x=nnz_B,
    #              y=gpu_9,
    #              s=dot_size,
    #              marker='o',
    #              color='blue',
    #              alpha=1,
    #              label='G_V3',
    #              # fontsize = 30
    #              )
    plt2.scatter(x=nnz_B,
                 y=gpu_10,
                 s=dot_size,
                 marker='o',
                 color=scatter_color[4],
                 alpha=1,
                 label='G_V3',
                 # fontsize = 30
                 )
    # plt2.scatter(x=nnz_B,
    #              y=gpu_11,
    #              s=dot_size,
    #              marker='o',
    #              color='cyan',
    #              alpha=1,
    #              label='G_V5',
    #              # fontsize = 30
    #              )

    plt2.tick_params(axis='both', labelsize=label_font_size)
    # plt2.xlim(0, 6.5)
    # plt2.set_title("SpTRSM", fontsize=label_font_size)
    plt2.set_xlabel("nnz_B ($\\log_{10}$)"+"\n"+"(c) GESSM",
                    fontsize=label_font_size)
    # plt2.set_ylabel("Time ($\\log_{10}$(ms))", fontsize=label_font_size)
    plt2.grid(c='black', alpha=grid_alpha, linestyle='--')
    plt2.ticklabel_format(style='plain', axis='x')
    plt2.legend(framealpha=0, ncol=3, loc='upper left',
                markerscale=mark_scale_size, fontsize=legend_size)


def main():
    fig, axs = plt.subplots(2, 2, dpi=300, figsize=(140, 110))
    plt.subplots_adjust(hspace=0.38, wspace=0.3)
    for ax in axs.flatten():
        for pos in ['top', 'bottom', 'left', 'right']:
            ax.spines[pos].set_linewidth(10)
    draw_GETRF(axs[0, 0])

    draw_TSTRF(axs[1, 0])
    draw_GESSM(axs[0, 1])
    draw_SSSSM(axs[1, 1])

    plt.savefig("../figure/figure7.pdf", bbox_inches='tight', dpi=300)
    # plt.show()


if __name__ == '__main__':
    main()
