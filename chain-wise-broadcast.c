#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    int rank, size, value;
    
    MPI_Init(&argc, &argv);  // Initialize MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // Get rank of the process
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // Get total number of processes

    if (rank == 0) {
        value = 100; // Initial value at rank 0
        printf("Process %d: Initial value = %d\n", rank, value);
    }

    // Chain-wise broadcasting
    if (rank > 0) {
        MPI_Recv(&value, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank < size - 1) {
        MPI_Send(&value, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    }

    // Print received value at each process
    printf("Process %d: Received value = %d\n", rank, value);

    MPI_Finalize();  // Finalize MPI
    return 0;
}

