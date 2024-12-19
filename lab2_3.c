//Задача B2
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define NUM_THREADS 4

double f(double x, double y) {
    return exp(y);
}

int main() {
    int n;
    double h, a = 1.0, b;
    int i, j, iter;
    FILE *fp;
    double start_time, end_time, cpu_time_used;

    fp = fopen("result_4.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "Ошибка открытия файла\n");
        return 1;
    }

    start_time = omp_get_wtime();

    for (n = 400; n <= 4000; n += 400) {
        h = 1.0 / (n - 1);
        for (b = 0.0; b <= 2.0; b += 0.1) { 
            double *y = (double *)malloc(n * sizeof(double));
            double *y_new = (double *)malloc(n * sizeof(double));
            double *x = (double *)malloc(n * sizeof(double));

            for (i = 0; i < n; i++) {
                x[i] = i * h;
                y[i] = a + (b - a) * x[i];
            }
            y[0] = a;
            y[n - 1] = b;

            double max_diff = 1.0;
            for (iter = 0; iter < 5000 && max_diff > 1e-6; iter++) { 
                #pragma omp parallel for num_threads(NUM_THREADS) private(i, j) shared(y, y_new, x, n, h)
                for (i = 1; i < n - 1; i++) {
                    double f_m = f(x[i], y[i]);
                    double f_p1 = f(x[i + 1], y[i + 1]);
                    double f_m1 = f(x[i - 1], y[i - 1]);

                    y_new[i] = (y[i + 1] + y[i - 1] - h * h * (f_m + (f_p1 - 2 * f_m + f_m1) / 12.0)) / 2.0;
                }
                max_diff = 0;
                for (i = 1; i < n - 1; i++) {
                    max_diff = fmax(max_diff, fabs(y_new[i] - y[i]));
                    y[i] = y_new[i];
                }
            }

            fprintf(fp, "n = %d, b = %.1f, y(0.5) = %f, iterations = %d, max_diff = %e\n", n, b, y[n / 2], iter, max_diff);

            free(y);
            free(y_new);
            free(x);
        }
    }

    fclose(fp);
    end_time = omp_get_wtime();
    cpu_time_used = end_time - start_time;
    printf("Время: %f секунд\n", cpu_time_used);

    return 0;
}

