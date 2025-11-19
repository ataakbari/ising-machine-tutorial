/*
 * Pure CUDA Implementation of Ising Machine with Simulated Annealing
 * Supports N-dimensional lattices with and without external field
 * 
 * Physics:
 *     Energy: E = -Σ J_ij s_i s_j - Σ h_i s_i
 *     where s_i ∈ {-1, +1} are spins
 *     J_ij: coupling between spins (interaction strength)
 *     h_i: external magnetic field (bias on each spin)
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)


/*
 * Initialize random number generator states (one per thread)
 */
__global__ void init_curand_states(curandState *states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}


/*
 * Initialize spins randomly to {-1, +1}
 */
__global__ void initialize_spins_kernel(float *spins, curandState *states, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Generate random float in [0, 1), convert to {-1, +1}
        float rand_val = curand_uniform(&states[idx]);
        spins[idx] = (rand_val < 0.5f) ? -1.0f : 1.0f;
    }
}


/*
 * Compute local field acting on spin i for nearest-neighbor lattice
 * 
 * For an N-dimensional lattice, sum over 2N neighbors (left/right in each dim)
 * h_eff = J * (sum of neighbor spins) + h_i
 */
__device__ float compute_local_field_nn(float *spins, float *h, int idx,
                                        int *dims, int ndim, float J) {
    float field = 0.0f;
    
    // Calculate strides for each dimension
    int strides[8];  // Support up to 8D (more than enough)
    strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; d--) {
        strides[d] = strides[d + 1] * dims[d + 1];
    }
    
    // Get multi-dimensional index
    int coords[8];
    int remaining = idx;
    for (int d = 0; d < ndim; d++) {
        coords[d] = remaining / strides[d];
        remaining %= strides[d];
    }
    
    // Sum over all 2N neighbors (forward and backward in each dimension)
    for (int d = 0; d < ndim; d++) {
        // Forward neighbor in dimension d
        int forward_coord = (coords[d] + 1) % dims[d];  // Periodic boundary
        int neighbor_idx = idx + (forward_coord - coords[d]) * strides[d];
        field += J * spins[neighbor_idx];
        
        // Backward neighbor in dimension d
        int backward_coord = (coords[d] - 1 + dims[d]) % dims[d];
        neighbor_idx = idx + (backward_coord - coords[d]) * strides[d];
        field += J * spins[neighbor_idx];
    }
    
    // Add external field if present
    if (h != nullptr) {
        field += h[idx];
    }
    
    return field;
}


/*
 * Metropolis Monte Carlo kernel for nearest-neighbor Ising model
 * 
 * Each thread attempts to flip one spin using Metropolis criterion:
 *   - Accept if ΔE < 0 (energy decreases)
 *   - Accept with probability exp(-ΔE/T) if ΔE > 0
 * 
 * Checkerboard decomposition: update even/odd sites separately to avoid conflicts
 */
__global__ void metropolis_step_kernel(float *spins, float *h, curandState *states,
                                       int *dims, int ndim, float J, float T,
                                       int parity, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // Checkerboard: only update if site matches current parity
    int sum_coords = 0;
    int remaining = idx;
    int stride = 1;
    for (int d = ndim - 1; d >= 0; d--) {
        int coord = (remaining / stride) % dims[d];
        sum_coords += coord;
        stride *= dims[d];
    }
    
    if ((sum_coords % 2) != parity) return;
    
    // Compute local field
    float h_eff = compute_local_field_nn(spins, h, idx, dims, ndim, J);
    
    // Energy change if we flip: ΔE = 2 * s_i * h_eff
    float delta_E = 2.0f * spins[idx] * h_eff;
    
    // Metropolis acceptance
    bool accept = false;
    if (delta_E < 0.0f) {
        accept = true;  // Always accept energy-lowering moves
    } else {
        // Accept with probability exp(-ΔE/T)
        float acceptance_prob = expf(-delta_E / T);
        float rand_val = curand_uniform(&states[idx]);
        accept = (rand_val < acceptance_prob);
    }
    
    if (accept) {
        spins[idx] *= -1.0f;  // Flip spin
    }
}


/*
 * Compute total energy (parallel reduction)
 */
__global__ void compute_energy_kernel(float *spins, float *h, int *dims, int ndim,
                                     float J, float *energy_out, int n) {
    __shared__ float shared_energy[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    float local_energy = 0.0f;
    
    if (idx < n) {
        // Interaction energy: -J * s_i * (sum of neighbors)
        // Each bond counted once by checking only forward neighbors
        int strides[8];
        strides[ndim - 1] = 1;
        for (int d = ndim - 2; d >= 0; d--) {
            strides[d] = strides[d + 1] * dims[d + 1];
        }
        
        int coords[8];
        int remaining = idx;
        for (int d = 0; d < ndim; d++) {
            coords[d] = remaining / strides[d];
            remaining %= strides[d];
        }
        
        // Only count forward neighbors to avoid double-counting
        for (int d = 0; d < ndim; d++) {
            int forward_coord = (coords[d] + 1) % dims[d];
            int neighbor_idx = idx + (forward_coord - coords[d]) * strides[d];
            local_energy += -J * spins[idx] * spins[neighbor_idx];
        }
        
        // External field energy: -h_i * s_i
        if (h != nullptr) {
            local_energy += -h[idx] * spins[idx];
        }
    }
    
    shared_energy[tid] = local_energy;
    __syncthreads();
    
    // Parallel reduction within block
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_energy[tid] += shared_energy[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result
    if (tid == 0) {
        atomicAdd(energy_out, shared_energy[0]);
    }
}


/*
 * Host function: Run simulated annealing
 */
void run_ising_annealing(float *spins_host, float *h_host, int *dims, int ndim,
                        float J, float T_initial, float T_final, int n_steps) {
    // Calculate total number of spins
    int n = 1;
    for (int d = 0; d < ndim; d++) {
        n *= dims[d];
    }
    
    printf("Running Ising annealing:\n");
    printf("  Total spins: %d\n", n);
    printf("  Dimensions: ");
    for (int d = 0; d < ndim; d++) {
        printf("%d%s", dims[d], (d < ndim - 1) ? "×" : "\n");
    }
    printf("  Coupling J: %.3f\n", J);
    printf("  Temperature: %.3f → %.3f\n", T_initial, T_final);
    printf("  Steps: %d\n\n", n_steps);
    
    // Allocate device memory
    float *d_spins, *d_h;
    int *d_dims;
    curandState *d_states;
    
    CUDA_CHECK(cudaMalloc(&d_spins, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dims, ndim * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_states, n * sizeof(curandState)));
    
    if (h_host != nullptr) {
        CUDA_CHECK(cudaMalloc(&d_h, n * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_h, h_host, n * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        d_h = nullptr;
    }
    
    CUDA_CHECK(cudaMemcpy(d_dims, dims, ndim * sizeof(int), cudaMemcpyHostToDevice));
    
    // Initialize random states
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    init_curand_states<<<blocks, threads>>>(d_states, time(NULL), n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Initialize spins randomly
    initialize_spins_kernel<<<blocks, threads>>>(d_spins, d_states, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Annealing loop
    for (int step = 0; step < n_steps; step++) {
        // Temperature schedule (exponential cooling)
        float progress = (float)step / (float)n_steps;
        float T = T_initial * powf(T_final / T_initial, progress);
        
        // Checkerboard updates (even then odd) to allow parallelism
        metropolis_step_kernel<<<blocks, threads>>>(d_spins, d_h, d_states,
                                                    d_dims, ndim, J, T, 0, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        metropolis_step_kernel<<<blocks, threads>>>(d_spins, d_h, d_states,
                                                    d_dims, ndim, J, T, 1, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Print progress every 1000 steps
        if (step % 1000 == 0) {
            printf("Step %d/%d, T=%.4f\n", step, n_steps, T);
        }
    }
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(spins_host, d_spins, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Compute final energy
    float *d_energy;
    float energy = 0.0f;
    CUDA_CHECK(cudaMalloc(&d_energy, sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_energy, &energy, sizeof(float), cudaMemcpyHostToDevice));
    
    compute_energy_kernel<<<blocks, threads>>>(d_spins, d_h, d_dims, ndim, J, d_energy, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&energy, d_energy, sizeof(float), cudaMemcpyDeviceToHost));
    
    printf("\nFinal energy: %.2f\n", energy);
    
    // Cleanup
    cudaFree(d_spins);
    cudaFree(d_dims);
    cudaFree(d_states);
    cudaFree(d_energy);
    if (d_h != nullptr) cudaFree(d_h);
}


/*
 * Example 1: 2D Ferromagnetic Ising (no external field)
 */
void example_2d_ferromagnetic() {
    printf("\n" "=========================================\n");
    printf("2D Ferromagnetic Ising Model\n");
    printf("=========================================\n\n");
    
    int dims[] = {64, 64};
    int ndim = 2;
    int n = dims[0] * dims[1];
    float J = 1.0f;  // Ferromagnetic
    
    float *spins = (float*)malloc(n * sizeof(float));
    float *h = nullptr;  // No external field
    
    run_ising_annealing(spins, h, dims, ndim, J, 5.0f, 0.01f, 5000);
    
    // Count magnetization
    float mag = 0.0f;
    for (int i = 0; i < n; i++) {
        mag += spins[i];
    }
    mag /= n;
    printf("Magnetization: %.3f\n", mag);
    
    free(spins);
}


/*
 * Example 2: 2D Antiferromagnetic Ising (no external field)
 */
void example_2d_antiferromagnetic() {
    printf("\n" "=========================================\n");
    printf("2D Antiferromagnetic Ising Model\n");
    printf("=========================================\n\n");
    
    int dims[] = {64, 64};
    int ndim = 2;
    int n = dims[0] * dims[1];
    float J = -1.0f;  // Antiferromagnetic
    
    float *spins = (float*)malloc(n * sizeof(float));
    float *h = nullptr;
    
    run_ising_annealing(spins, h, dims, ndim, J, 5.0f, 0.01f, 5000);
    
    free(spins);
}


/*
 * Example 3: 2D Ising with external field
 */
void example_2d_with_field() {
    printf("\n" "=========================================\n");
    printf("2D Ising Model WITH External Field\n");
    printf("=========================================\n\n");
    
    int dims[] = {64, 64};
    int ndim = 2;
    int n = dims[0] * dims[1];
    float J = 1.0f;
    float h_strength = 0.5f;
    
    float *spins = (float*)malloc(n * sizeof(float));
    float *h = (float*)malloc(n * sizeof(float));
    
    // Uniform external field
    for (int i = 0; i < n; i++) {
        h[i] = h_strength;
    }
    
    printf("External field strength: %.3f\n\n", h_strength);
    run_ising_annealing(spins, h, dims, ndim, J, 5.0f, 0.01f, 5000);
    
    // Magnetization
    float mag = 0.0f;
    for (int i = 0; i < n; i++) {
        mag += spins[i];
    }
    mag /= n;
    printf("Magnetization: %.3f (should be biased toward +1)\n", mag);
    
    free(spins);
    free(h);
}


/*
 * Example 4: 3D Ising model
 */
void example_3d_ferromagnetic() {
    printf("\n" "=========================================\n");
    printf("3D Ferromagnetic Ising Model\n");
    printf("=========================================\n\n");
    
    int dims[] = {32, 32, 32};
    int ndim = 3;
    int n = dims[0] * dims[1] * dims[2];
    float J = 1.0f;
    
    float *spins = (float*)malloc(n * sizeof(float));
    float *h = nullptr;
    
    run_ising_annealing(spins, h, dims, ndim, J, 5.0f, 0.01f, 5000);
    
    free(spins);
}


int main() {
    printf("CUDA Ising Machine Examples\n");
    printf("============================\n");
    
    // Run examples
    example_2d_ferromagnetic();
    example_2d_antiferromagnetic();
    example_2d_with_field();
    example_3d_ferromagnetic();
    
    printf("\n" "============================\n");
    printf("All examples completed!\n");
    printf("============================\n");
    
    return 0;
}

