#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define DATA_SIZE 1000000   // Total number of data points
#define VALUE_RANGE 100      // Range of values (0-99)
#define NUM_PROCESSES 4     // Number of MPI processes

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != NUM_PROCESSES) {
        if (rank == 0) {
            printf("This program requires exactly %d processes.\n", NUM_PROCESSES);
        }
        MPI_Finalize();
        return 1;
    }

    int local_data_size = DATA_SIZE / NUM_PROCESSES;
    int *data = NULL;
    int *local_data = (int *)malloc(local_data_size * sizeof(int));
    // Each process creates a full-range histogram for its local data
    int *local_hist = (int *)calloc(VALUE_RANGE, sizeof(int));
    int *final_hist = NULL;

    if (rank == 0) {
        data = (int *)malloc(DATA_SIZE * sizeof(int));
        srand(time(NULL));
        for (int i = 0; i < DATA_SIZE; i++) {
            data[i] = rand() % VALUE_RANGE;
        }
    }

    MPI_Scatter(data, local_data_size, MPI_INT, local_data, local_data_size, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        free(data);
    }

    // Each process builds a complete histogram for its own data chunk
    for (int i = 0; i < local_data_size; i++) {
        int value = local_data[i];
        local_hist[value]++; // Count every value in the local chunk
    }
    free(local_data); // Free the local data as we don't need it anymore

    if (rank == 0) {
        final_hist = (int *)calloc(VALUE_RANGE, sizeof(int));
    }

    // Use MPI_Reduce to sum all partial histograms into final_hist on rank 0.
    MPI_Reduce(local_hist, final_hist, VALUE_RANGE, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    free(local_hist);

    if (rank == 0) {
        printf("Final Histogram:\n");
        for (int i = 0; i < VALUE_RANGE; i++) {
            printf("Value %2d: %d\n", i, final_hist[i]);
        }
        free(final_hist);
    }

    MPI_Finalize();
    return 0;
}
