#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 800
#define HEIGHT 800
#define MAX_ITER 10000

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;
    start_time = MPI_Wtime();

    double real_min = -2.0, real_max = 1.0;
    double imag_min = -1.5, imag_max = 1.5;

    int chunk_size = HEIGHT / size;
    int start_row = rank * chunk_size;
    int end_row = start_row + chunk_size;
    if (rank == size - 1) {
        end_row = HEIGHT;
    }

    int* image = (int*) malloc(WIDTH * chunk_size * sizeof(int));
    for (int y = start_row; y < end_row; y++) {
        for (int x = 0; x < WIDTH; x++) {
            double c_real = real_min + x * (real_max - real_min) / WIDTH;
            double c_imag = imag_min + y * (imag_max - imag_min) / HEIGHT;
            double z_real = 0, z_imag = 0;
            int i;
            for (i = 0; i < MAX_ITER; i++) {
                double z_real_temp = z_real * z_real - z_imag * z_imag + c_real;
                double z_imag_temp = 2 * z_real * z_imag + c_imag;
                z_real = z_real_temp;
                z_imag = z_imag_temp;
                if (z_real * z_real + z_imag * z_imag > 4) {
                    break;
                }
            }
            image[(y - start_row) * WIDTH + x] = i;
        }
    }

    if (rank == 0) {
        int* recv_buffer = (int*) malloc(WIDTH * HEIGHT * sizeof(int));
        for (int i = 0; i < chunk_size * WIDTH; i++) {
            recv_buffer[i] = image[i];
        }
        for (int i = 1; i < size; i++) {
            MPI_Recv(recv_buffer + i * chunk_size * WIDTH, chunk_size * WIDTH, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        FILE* file = fopen("mandelbrot.pgm", "wb");
        fprintf(file, "P2\n%d %d\n%d\n", WIDTH, HEIGHT, MAX_ITER);
        for (int y = 0; y < HEIGHT; y++) {
            for (int x = 0; x < WIDTH; x++) {
                fprintf(file, "%d ", recv_buffer[y * WIDTH + x]);
            }
            fprintf(file, "\n");
        }
        fclose(file);
        free(recv_buffer);
    } else {
        MPI_Send(image, chunk_size * WIDTH, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    free(image);

    end_time = MPI_Wtime();
    printf("Execution time = %f seconds\n", end_time - start_time);

    MPI_Finalize();
    return 0;
}