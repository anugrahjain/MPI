/* parallel-matrix-mul-mpi.c
   MPI-only version for parallel computation + timing.
   Uses MPI_Scatter, MPI_Bcast, MPI_Gather.
   Timing measured from just before scatter to just after gather.
*/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

/* Allocate an NxN matrix as row pointers + contiguous block */
int **allocate_matrix(int N) {
    int **mat = (int **) malloc(N * sizeof(int *));
    int *data = (int *) malloc(N * N * sizeof(int));
    for (int i = 0; i < N; i++) mat[i] = data + i * N;
    return mat;
}
void free_matrix(int **mat) {
    free(mat[0]);
    free(mat);
}

/* Utility to fill matrix (same logic as in your sequential code) */
void fill_matrix(int **M, int N) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            M[i][j] = (i + 1) * (j + 1);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 1500;                   /* default size */
    if (argc >= 2) N = atoi(argv[1]);
    if (N % size != 0) {
        if (rank == 0) fprintf(stderr, "N must be divisible by %d\n", size);
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = N / size;

    int **A = NULL, **B = NULL, **C_par = NULL;
    int *A_data = NULL, *B_data = NULL, *C_data = NULL;

    if (rank == 0) {
        A = allocate_matrix(N);
        B = allocate_matrix(N);
        C_par = allocate_matrix(N);

        fill_matrix(A, N);
        fill_matrix(B, N);

        A_data = A[0];
        B_data = B[0];
        C_data = C_par[0];
    } else {
        B = allocate_matrix(N);
        B_data = B[0];
    }

    /* Local buffers */
    int *local_A_data = (int *) malloc(rows_per_proc * N * sizeof(int));
    int *local_C_data = (int *) malloc(rows_per_proc * N * sizeof(int));

    int **local_A = (int **) malloc(rows_per_proc * sizeof(int *));
    int **local_C = (int **) malloc(rows_per_proc * sizeof(int *));
    for (int i = 0; i < rows_per_proc; ++i) {
        local_A[i] = local_A_data + i * N;
        local_C[i] = local_C_data + i * N;
    }

    /* Synchronize and start timer */
    MPI_Barrier(MPI_COMM_WORLD);
    clock_t tstart_par = clock();

    /* Scatter rows of A, broadcast B */
    MPI_Scatter(rank == 0 ? A_data : NULL, rows_per_proc * N, MPI_INT,
                local_A_data, rows_per_proc * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(B_data, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    /* Local multiplication */
    for (int i = 0; i < rows_per_proc; ++i) {
        for (int j = 0; j < N; ++j) {
            local_C[i][j] = 0;
            for (int k = 0; k < N; ++k) {
                local_C[i][j] += local_A[i][k] * B[k][j];
            }
        }
    }

    /* Gather result */
    MPI_Gather(local_C_data, rows_per_proc * N, MPI_INT,
               rank == 0 ? C_data : NULL, rows_per_proc * N, MPI_INT,
               0, MPI_COMM_WORLD);

    /* Stop timer */
    clock_t tend_par = clock();
    double time_par = (double)(tend_par - tstart_par) / CLOCKS_PER_SEC;

    /* Report maximum time across all ranks */
    double max_time_par;
    MPI_Reduce(&time_par, &max_time_par, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Parallel MPI time (scatter+bcast+compute+gather): %f seconds\n", max_time_par);
    }

    /* Cleanup */
    free(local_A_data); free(local_C_data);
    free(local_A); free(local_C);
    if (rank == 0) {
        free_matrix(A); free_matrix(B); free_matrix(C_par);
    } else {
        free_matrix(B);
    }

    MPI_Finalize();
    return 0;
}

