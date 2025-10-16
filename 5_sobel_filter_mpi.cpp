#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Sobel Filter Function
void apply_sobel_filter(unsigned char* input, unsigned char* output, int rows, int cols) {
    // Sobel Kernels
    int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};  // vertical mask
    int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};  // horizontal mask

    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            int sumX = 0, sumY = 0;

            // Apply Sobel operator
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    int pixel = input[(i + ki) * cols + (j + kj)];
                    sumX += pixel * Gx[ki + 1][kj + 1];
                    sumY += pixel * Gy[ki + 1][kj + 1];
                }
            }
            // Compute gradient magnitude
            int magnitude = sqrt(sumX * sumX + sumY * sumY);
            output[i * cols + j] = (magnitude > 255) ? 255 : magnitude; // Normalize
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = 0, cols = 0, chunk_size = 0;
    unsigned char *image = NULL, *local_chunk = NULL, *filtered_chunk = NULL, *filtered_image = NULL;

    if (rank == 0) {
        // Load Grayscale image
        Mat gray_img = imread("image.jpg", IMREAD_GRAYSCALE);
        if (gray_img.empty()) {
            cout << "Error: Could not open grayscale image" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        rows = gray_img.rows;
        cols = gray_img.cols;

        if (rows % size != 0) {
            cout << "Error: Number of rows must be divisible by number of processes" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        chunk_size = rows / size;
        image = new unsigned char[rows * cols];  // Proper memory allocation
        memcpy(image, gray_img.data, rows * cols);  // Copy image data
    }

    // Broadcast image dimensions
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
    chunk_size = rows / size;

    // Allocate memory for each process
    local_chunk = new unsigned char[chunk_size * cols]();
    filtered_chunk = new unsigned char[chunk_size * cols]();

    // Scatter image data
    MPI_Scatter(image, chunk_size * cols, MPI_UNSIGNED_CHAR,
                local_chunk, chunk_size * cols, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Apply Sobel filter
    apply_sobel_filter(local_chunk, filtered_chunk, chunk_size, cols);

    // Gather processed image data
    if (rank == 0) {
        filtered_image = new unsigned char[rows * cols]();  // Proper memory allocation
    }

    MPI_Gather(filtered_chunk, chunk_size * cols, MPI_UNSIGNED_CHAR,
               filtered_image, chunk_size * cols, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Save final image
    if (rank == 0) {
        Mat output(rows, cols, CV_8UC1, filtered_image);
        imwrite("sobel_filtered_image.jpg", output);
        cout << "Image processing complete! Sobel-filtered image saved." << endl;
        delete[] filtered_image;
    }

    // Free memory
    delete[] local_chunk;
    delete[] filtered_chunk;
    if (rank == 0) delete[] image;

    MPI_Finalize();
    return 0;
}

