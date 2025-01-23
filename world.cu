#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "world.h"

#define __CUDACC__
#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <curand_kernel.h>
#define CUDA_SM 128
#define CUDA_WARP_SIZE 32
#define CUDA_BLOCK_DIM_X 16
#define CUDA_BLOCK_DIM_Y 16
#define CUDA_NB_THREAD (CUDA_BLOCK_DIM_X * CUDA_BLOCK_DIM_Y)
#define CUDA_NB_BLOCK (CUDA_SM / CUDA_WARP_SIZE)


__global__ void setup_kernel(curandState* state, uint64_t seed) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    int index = i * gridDim.x * blockDim.x + j;
    curand_init(seed, index, 0, &state[index]);
}
__global__ void generate_randoms(curandState* globalState, float* randoms) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    int index = i * gridDim.x * blockDim.x + j;
    curandState localState = globalState[index];  
    randoms[index] = curand_uniform(&localState);
    globalState[index] = localState;
}

static inline __device__ bool cuda_should_happen(int probability, curandState *state, int threadIdX, int threadIdY) 
{
    return probability < (curand(&state[threadIdX + threadIdY]) % 100);
}

static __device__ uint8_t cuda_world_get_nb_infected_neighbours(const world_t *p, size_t i,
						size_t j)
{
	uint8_t sum = 0;
	for (int dx = -p->params.proximity; dx <= p->params.proximity; ++dx) {
		for (int dy = -p->params.proximity; dy <= p->params.proximity;
		     ++dy) {
			if (dx == 0 && dy == 0) {
				continue;
			}
			const size_t ni = i + dx;
			const size_t nj = j + dy;
			if (!(ni < p->params.worldHeight &&
			      nj < p->params.worldWidth)) {
				continue;
			}

			sum += p->grid[ni * p->params.worldWidth + nj] ==
			       INFECTED;
		}
	}
	return sum;
}

bool __device__ cuda_world_should_infect(const world_t *p, curandState *state, size_t i, size_t j, int probability)
{
	return cuda_world_get_nb_infected_neighbours(p, i, j) &&
	       cuda_should_happen(probability,state ,i, j);
}
void __device__ cuda_world_infect_if_should_infect(const world_t *p, state_t *world, curandState *state, size_t i,
				   size_t j, int probability)
{
	if (cuda_world_should_infect(p,state, i, j, probability)) {
		world[i * p->params.worldWidth + j] = INFECTED;
	}
}
void __device__ cuda_world_handle_infected(world_t *p, state_t *world, curandState *state, size_t i, size_t j)
{
	const size_t index = i * p->params.worldWidth + j;

	if (p->infectionDurationGrid[index] == 0) {
		if (cuda_should_happen(p->params.deathProbability, state, i, j)) {
			world[index] = DEAD;
		} else {
			world[index] = IMMUNE;
			p->infectionDurationGrid[index] =
				p->params.infectionDuration;
		}
	} else {
		p->infectionDurationGrid[index]--;
	}
}

static __global__ void cuda_world_update(world_t *d_p_in, state_t *d_tmp_world, curandState *state)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < d_p_in->params.worldHeight && j < d_p_in->params.worldWidth) {
        int index = i * d_p_in->params.worldWidth + j;
        switch (d_p_in->grid[index]) {
            case HEALTHY:
                cuda_world_infect_if_should_infect(d_p_in, d_tmp_world, state, i, j, d_p_in->params.healthyInfectionProbability);
                break;
            case IMMUNE:
                cuda_world_infect_if_should_infect(d_p_in, d_tmp_world, state, i, j, d_p_in->params.immuneInfectionProbability);
                break;
            case INFECTED:
                cuda_world_handle_infected(d_p_in, d_tmp_world, state, i, j);
                break;
            case EMPTY:
            case DEAD:
                break;
        }
    }
    size_t index = i * d_p_in->params.worldWidth + j;
    d_p_in->grid[index] = d_tmp_world[index];
}

#endif

static inline size_t world_world_size(const world_parameters_t *p)
{
	return p->worldWidth * p->worldHeight;
}


static inline bool should_happen(int probability)
{
	return probability < (rand() % 100);
}

static size_t get_in_state(const world_t *p, state_t state)
{
	const size_t size = world_world_size(&p->params);
	size_t sum = 0;
	for (size_t i = 0; i < size; ++i) {
		sum += p->grid[i] == state;
	}
	return sum;
}
static size_t world_initial_population(const world_parameters_t *p)
{
	return world_world_size(p) * p->populationPercent / 100;
}

static uint8_t world_get_nb_infected_neighbours(const world_t *p, size_t i,
						size_t j)
{
	uint8_t sum = 0;
	for (int dx = -p->params.proximity; dx <= p->params.proximity; ++dx) {
		for (int dy = -p->params.proximity; dy <= p->params.proximity;
		     ++dy) {
			if (dx == 0 && dy == 0) {
				continue;
			}
			const size_t ni = i + dx;
			const size_t nj = j + dy;
			if (!(ni < p->params.worldHeight &&
			      nj < p->params.worldWidth)) {
				continue;
			}

			sum += p->grid[ni * p->params.worldWidth + nj] ==
			       INFECTED;
		}
	}
	return sum;
}

int world_init(world_t *world, const world_parameters_t *p)
{
	if (!world) {
		return -1;
	}
	const size_t world_size = world_world_size(p);
	const size_t people_to_spawn = world_initial_population(p);

	if (!world_size) {
		return -1;
	}
	if (!(people_to_spawn >= p->initialImmune + p->initialInfected)) {
		return -1;
	}

	world->grid = (state_t*)malloc(world_size * sizeof(*world->grid));

	world->infectionDurationGrid =
		(uint8_t*)malloc(world_size * sizeof(*world->infectionDurationGrid));

	if (!world->grid || !world->infectionDurationGrid) {
		return -1;
	}
	memset(world->grid, EMPTY, sizeof(*world->grid) * world_size);
	memset(world->infectionDurationGrid, p->infectionDuration,
	       sizeof(*world->infectionDurationGrid) * world_size);
	memcpy(&world->params, p, sizeof(*p));

	srand(time(NULL));

	size_t people = 0;
	size_t people_infected = 0;
	size_t people_immune = 0;

	while (people < people_to_spawn) {
		const size_t i = rand() % world_size;
		if (world->grid[i] != EMPTY) {
			continue;
		}
		++people;
		if (people_infected < p->initialInfected) {
			world->grid[i] = INFECTED;
			++people_infected;
			continue;
		}
		if (people_immune < p->initialImmune) {
			world->grid[i] = IMMUNE;
			++people_immune;
			continue;
		}
		world->grid[i] = HEALTHY;
	}
	return 0;
}

size_t world_get_infected(const world_t *p)
{
	return get_in_state(p, INFECTED);
}

size_t world_get_healthy(const world_t *p)
{
	return get_in_state(p, HEALTHY);
}

size_t world_get_immune(const world_t *p)
{
	return get_in_state(p, IMMUNE);
}

size_t world_get_dead(const world_t *p)
{
	return get_in_state(p, DEAD);
}

bool world_should_infect(const world_t *p, size_t i, size_t j, int probability)
{
	return world_get_nb_infected_neighbours(p, i, j) &&
	       should_happen(probability);
}
void world_infect_if_should_infect(const world_t *p, state_t *world, size_t i,
				   size_t j, int probability)
{
	if (world_should_infect(p, i, j, probability)) {
		world[i * p->params.worldWidth + j] = INFECTED;
	}
}
void world_handle_infected(world_t *p, state_t *world, size_t i, size_t j)
{
	const size_t index = i * p->params.worldWidth + j;

	if (p->infectionDurationGrid[index] == 0) {
		if (should_happen(p->params.deathProbability)) {
			world[index] = DEAD;
		} else {
			world[index] = IMMUNE;
			p->infectionDurationGrid[index] =
				p->params.infectionDuration;
		}
	} else {
		p->infectionDurationGrid[index]--;
	}
}

void *world_prepare_update(const world_t *p)
{
	return calloc(world_world_size(&p->params), sizeof(*p->grid));
}

static void world_update_simple(world_t *p, state_t *tmp_world)
{
	for (size_t i = 0; i < p->params.worldHeight; i++) {
		for (size_t j = 0; j < p->params.worldWidth; j++) {
			switch (p->grid[i * p->params.worldWidth + j]) {
			case HEALTHY:
				world_infect_if_should_infect(
					p, tmp_world, i, j,
					p->params.healthyInfectionProbability);
				break;
			case IMMUNE:
				world_infect_if_should_infect(
					p, tmp_world, i, j,
					p->params.immuneInfectionProbability);
				break;

			case INFECTED:
				world_handle_infected(p, tmp_world, i, j);
				break;
			case EMPTY:
			case DEAD:
				break;
			}
		}
	}
}

void world_update(world_t *p, void *raw)
{
	const size_t world_size = world_world_size(&p->params);
	state_t *tmp_world = (state_t *)raw;

	memcpy(tmp_world, p->grid, world_size * sizeof(*p->grid));
    printf("Debug line : %d passed\n", __LINE__);

#ifdef __CUDACC__
    world_t* d_p_in;
    state_t* d_tmp_world;
    cudaError_t err;
    curandState* dev_curand_states;
    float* randomValues;
    printf("Debug line : %d passed\n", __LINE__);

    // Allocate memory for the world struct and its members
    cudaMalloc(&d_p_in, sizeof(world_t));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error allocating memory for d_p_in\n");
        exit(1);
    }
    cudaMalloc(&(d_p_in->grid), world_size * sizeof(state_t));
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error allocating memory for d_p_in->grid\n");
        exit(1);
    }

    cudaMalloc(&(d_p_in->infectionDurationGrid), world_size * sizeof(uint8_t));
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("Error allocating memory for d_p_in->infectionDurationGrid\n");
        exit(1);
    }

    cudaMalloc(&d_tmp_world, world_size * sizeof(state_t));
    err = cudaGetLastError();

    if (err != cudaSuccess) {
        printf("Error allocating memory for d_tmp_world\n");
        exit(1);
    }

    printf("Debug line : %d passed\n", __LINE__);

    // Allocate memory for random number generator
    cudaMalloc(&dev_curand_states, CUDA_NB_THREAD * sizeof(curandState));
    cudaMalloc(&randomValues, CUDA_NB_THREAD * sizeof(float));
    printf("Debug line : %d passed\n", __LINE__);

    // Copy data to GPU
    cudaMemcpy(d_p_in->grid, p->grid, world_size * sizeof(state_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p_in->infectionDurationGrid, p->infectionDurationGrid, world_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(&(d_p_in->params), &p->params, sizeof(p->params), cudaMemcpyHostToDevice);
    printf("Debug line : %d passed\n", __LINE__);

    dim3 blockDim(CUDA_BLOCK_DIM_X, CUDA_BLOCK_DIM_Y);
    dim3 gridDim((p->params.worldWidth + blockDim.x - 1) / blockDim.x, 
                (p->params.worldHeight + blockDim.y - 1) / blockDim.y);
    printf("Debug line : %d passed\n", __LINE__);

    setup_kernel<<<gridDim, blockDim>>>(dev_curand_states, time(NULL));
    cudaDeviceSynchronize();
    printf("Debug line : %d passed\n", __LINE__);

    generate_randoms<<<gridDim, blockDim>>>(dev_curand_states, randomValues);
    cudaDeviceSynchronize();
    printf("Debug line : %d passed\n", __LINE__);

    cuda_world_update<<<gridDim, blockDim>>>(d_p_in, d_tmp_world, dev_curand_states);
    cudaDeviceSynchronize();
    printf("Debug line : %d passed\n", __LINE__);

    
    cudaMemcpy(p->grid, d_p_in->grid, world_size * sizeof(*p->grid), cudaMemcpyDeviceToHost);
    printf("Debug line : %d passed\n", __LINE__);


    // Free GPU memory
    cudaFree(d_p_in->grid);
    cudaFree(d_p_in->infectionDurationGrid);
    cudaFree(d_p_in);
    cudaFree(d_tmp_world);
    cudaFree(dev_curand_states);
    cudaFree(randomValues);
    printf("Debug line : %d passed\n", __LINE__);


#else
	world_update_simple(p, tmp_world);
#endif

	memcpy(p->grid, tmp_world, world_size * sizeof(*p->grid));
}

void world_destroy(world_t *w)
{
	free(w->grid);
	free(w->infectionDurationGrid);
}
