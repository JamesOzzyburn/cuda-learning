#include <stdio.h>
#include <vector>
#include <ranges>

__global__ void vecAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n {1000000};
    size_t bytes {n * sizeof(float)};

    std::vector<float> h_a(n), h_b(n), h_c(n);

    for (const int i : std::views::iota(0, n)) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    vecAdd<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);

    printf("c[0] = %f expected 0\n", h_c[0]);
    printf("c[999999] = %f expected 2999997\n", h_c[999999]);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
