#ifndef PANGULU_TIME_H
#define PANGULU_TIME_H

#include "pangulu_common.h"

void pangulu_time_check_begin(struct timeval *GET_TIME_START)
{
    gettimeofday((GET_TIME_START), NULL);
}

double pangulu_time_check_end(struct timeval *GET_TIME_START)
{
    struct timeval GET_TIME_END;
    gettimeofday((&GET_TIME_END), NULL);
    return (((GET_TIME_END.tv_sec - GET_TIME_START->tv_sec) * 1000.0 + (GET_TIME_END.tv_usec - GET_TIME_START->tv_usec) / 1000.0))/1000.0;
}

void pangulu_time_init()
{
    TIME_transport = 0.0;
    TIME_isend = 0.0;
    TIME_receive = 0.0;
    TIME_getrf = 0.0;
    TIME_tstrf = 0.0;
    TIME_gessm = 0.0;
    TIME_gessm_sparse = 0.0;
    TIME_gessm_dense = 0.0;
    TIME_ssssm = 0.0;
    TIME_cuda_memcpy = 0.0;
    TIME_wait = 0.0;
    return;
}
void pangulu_time_output()
{
    // printf("rank\ttransport\tTIME_isend\tTIME_receive\tTIME_getrf\tTIME_tstrf\tTIME_gessm\tTIME_ssssm\tTIME_sum\tTIME_cudampy\tTIME_wait\n");
    printf("rank\tgetrf\ttstrf\tgessm\tssssm\tcal_sum\n");
}

void pangulu_time_simple_output(int_t rank)
{
    printf("%ld\t%.5lf\t%.5lf\t%.5lf\t%.5lf\t%.5lf\t\n",
           rank,
           TIME_getrf,
           TIME_tstrf,
           TIME_gessm,
           //    TIME_gessm_sparse,
           TIME_ssssm, TIME_gessm+TIME_getrf + TIME_tstrf + TIME_ssssm);
    // printf("sparse time :%.3lf\ndense time :%.3lf\ntotal time :%.3lf\n", TIME_gessm_sparse, TIME_gessm_dense, TIME_gessm_sparse + TIME_gessm_dense);
}

#ifdef VISUALISE
#include <mpi.h>
void pangulu_time_check_begin_visual()
{
    gettimeofday(&(GET_TIME_visual_start), NULL);
}

double pangulu_get_time_begin_visual()
{
    return GET_TIME_visual_start.tv_sec * 1000.0 + GET_TIME_visual_start.tv_usec / 1000.0;
}

double pangulu_get_time_end_visual()
{
    return GET_TIME_visual_end.tv_sec * 1000.0 + GET_TIME_visual_end.tv_usec / 1000.0;
}

double pangulu_time_check_end_visual()
{
    gettimeofday(&(GET_TIME_visual_end), NULL);
    return ((GET_TIME_visual_end.tv_sec - GET_TIME_visual_start.tv_sec) * 1000.0 + (GET_TIME_visual_end.tv_usec - GET_TIME_visual_start.tv_usec) / 1000.0);
}

void pangulu_init_visual_file(char *filename)
{
    visual_fp = NULL;
    char log_name[512];
    int_t file_length = strlen(filename);
    strcpy(log_name, filename);
    int_32t t = RANK;
    log_name[file_length - 2] = t % 10 + '0';
    t /= 10;
    log_name[file_length - 3] = t % 10 + '0';
    t /= 10;
    log_name[file_length - 4] = t % 10 + '0';
    log_name[file_length - 1] = '.';
    log_name[file_length] = 'c';
    log_name[file_length + 1] = 's';
    log_name[file_length + 2] = 'v';
    log_name[file_length + 3] = 0;

    visual_fp = fopen(log_name, "w+");

    if (visual_fp == NULL)
    {
        printf("error init visual log\n");
        exit(0);
    }
}

void pangulu_stop_visual_file()
{
    fclose(visual_fp);
    visual_fp = NULL;
}

#endif

#endif