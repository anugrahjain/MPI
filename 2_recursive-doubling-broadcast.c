#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, value;
    
    MPI_Init(&argc, &argv);  // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    if (size != 16) {
        if (rank == 0) {
            printf("This program requires exactly 16 MPI processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    // Initialize value at rank 0
    if (rank == 0) {
        value = 100; 
        printf("Process %d: Initial value = %d\n", rank, value);
    }

    // Recursive Doubling Broadcast using Hypercube Network
    int step;
    for (step = 1; step < size; step *= 2) {
        if (rank < step && rank + step < size) {
            MPI_Send(&value, 1, MPI_INT, rank + step, 0, MPI_COMM_WORLD);
        }
        if (rank >= step && rank < 2 * step) {
            MPI_Recv(&value, 1, MPI_INT, rank - step, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Print received value at each process
    printf("Process %d: Received value = %d\n", rank, value);

    MPI_Finalize();  // Finalize MPI
    return 0;
}

