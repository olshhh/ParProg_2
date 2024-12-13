#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <N>\n", argv[0]);
        return 1;
    }

    long long N = atoll(argv[1]);
    long long sum = 0;
    long long partial_sums[omp_get_max_threads()];

    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        long long thread_sum = 0;
        long long start = (long long)thread_id * (N / omp_get_max_threads());
        long long end = (thread_id == omp_get_max_threads() - 1) ? N : (thread_id + 1) * (N / omp_get_max_threads());


        for (long long i = start; i < end; i++) {
            thread_sum += i;
        }
        partial_sums[thread_id] = thread_sum;
    }

    for (int i = 0; i < omp_get_max_threads(); i++) {
        sum += partial_sums[i];
    }

    printf("Sum from 1 to %lld is: %lld\n", N, sum);
    return 0;
}

