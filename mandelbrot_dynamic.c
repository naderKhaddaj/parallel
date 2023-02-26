#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define WIDTH 800
#define HEIGHT 800
#define MAX_ITER 1000

int main(int argc, char **argv) {
    int rank, size, x, y, i, j, iter, chunk_size;
    double cx, cy, zx, zy, zx_new, x_min, x_max, y_min, y_max, dx, dy;
    int *image, *chunk;
    MPI_Status status;

    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time, end_time;
    start_time = MPI_Wtime();
    
    chunk_size = ceil((double)HEIGHT / size);
    x_min = -2.0;
    x_max = 1.0;
    y_min = -1.5;
    y_max = 1.5;
    dx = (x_max - x_min) / WIDTH;
    dy = (y_max - y_min) / HEIGHT;
    
    chunk = (int *)malloc(chunk_size * WIDTH * sizeof(int));
    
    if (rank == 0) {
        image = (int *)malloc(WIDTH * HEIGHT * sizeof(int));
    }
    
    for (i = rank * chunk_size; i < (rank + 1) * chunk_size && i < HEIGHT; i++) {
        for (j = 0; j < WIDTH; j++) {
            cx = x_min + j * dx;
            cy = y_min + i * dy;
            zx = 0.0;
            zy = 0.0;
            iter = 0;
            
            while (iter < MAX_ITER && (zx * zx + zy * zy) < 4.0) {
                zx_new = zx * zx - zy * zy + cx;
                zy = 2.0 * zx * zy + cy;
                zx = zx_new;
                iter++;
            }
            
            if (iter == MAX_ITER) {
                chunk[(i - rank * chunk_size) * WIDTH + j] = 0;
            } else {
                chunk[(i - rank * chunk_size) * WIDTH + j] = iter;
            }
        }
    }
    
    MPI_Gather(chunk, chunk_size * WIDTH, MPI_INT, image, chunk_size * WIDTH, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        FILE *fp = fopen("mandelbrot_dynamic.ppm", "wb");
        fprintf(fp, "P6\n%d %d\n255\n", WIDTH, HEIGHT);
        
        for (i = 0; i < HEIGHT; i++) {
            for (j = 0; j < WIDTH; j++) {
                iter = image[i * WIDTH + j];
                
                if (iter == MAX_ITER) {
                    fputc(0, fp);
                    fputc(0, fp);
                    fputc(0, fp);
                } else {
                    fputc((iter * 16) % 256, fp);
                    fputc((iter * 16) % 256, fp);
                    fputc((iter * 16) % 256, fp);
                }
            }
        }
        
        fclose(fp);
        free(image);
    }
    
    free(chunk);

    end_time = MPI_Wtime();
    printf("Execution time = %f seconds\n", end_time - start_time);

    MPI_Finalize();

    return 0;
}