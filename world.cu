#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include "world.h"
#include <cuda_runtime.h>

#define CUDA_SM 128
#define CUDA_WARP_SIZE 32

#define CUDA_NB_BLOCK (CUDA_SM / CUDA_WARP_SIZE)

static inline size_t world_world_size(const world_parameters_t *p)
{
	return p->worldWidth * p->worldHeight;
}
static inline __device__ bool should_happen(int probability)
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

bool __device__ world_should_infect(const world_t *p, size_t i, size_t j, int probability)
{
	return world_get_nb_infected_neighbours(p, i, j) &&
	       should_happen(probability);
}
void __device__ world_infect_if_should_infect(const world_t *p, state_t *world, size_t i,
				   size_t j, int probability)
{
	if (world_should_infect(p, i, j, probability)) {
		world[i * p->params.worldWidth + j] = INFECTED;
	}
}
void __device__ world_handle_infected(world_t *p, state_t *world, size_t i, size_t j)
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
static __global__ void world_update_cuda(world_t *d_p_in, world_t *d_p_out,
					 state_t *d_tmp_world)
{
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < d_p_in->params.worldHeight && j < d_p_in->params.worldWidth) {
        int index = i * d_p_in->params.worldWidth + j;
        
        switch (d_p_in->grid[index]) {
            case HEALTHY:
                world_infect_if_should_infect(d_p_in, d_tmp_world, i, j, d_p_in->params.healthyInfectionProbability);
                break;
            case IMMUNE:
                world_infect_if_should_infect(d_p_in, d_tmp_world, i, j, d_p_in->params.immuneInfectionProbability);
                break;
            case INFECTED:
                world_handle_infected(d_p_in, d_tmp_world, i, j);
                break;
            case EMPTY:
            case DEAD:
                break;
        }
    }
}
void world_update(world_t *p, void *raw)
{
	const size_t world_size = world_world_size(&p->params);
	state_t *tmp_world = (state_t *)raw;

	memcpy(tmp_world, p->grid, world_size * sizeof(*p->grid));

#ifdef __CUDACC__

world_t *d_p_in, *d_p_out;
state_t *d_tmp_world;

cudaMalloc(&d_p_in, sizeof(world_t));
cudaMalloc(&d_p_out, sizeof(world_t));
cudaMalloc(&d_tmp_world, world_size * sizeof(*tmp_world));

// Copy the CPU structure to a temporary copy on the CPU
world_t h_p_in = *p;

// Allocate arrays on the GPU
cudaMalloc(&h_p_in.grid, world_size * sizeof(*p->grid));
cudaMalloc(&h_p_in.infectionDurationGrid, world_size * sizeof(*p->infectionDurationGrid));

// Copy the array data to the GPU
cudaMemcpy(h_p_in.grid, p->grid, world_size * sizeof(*p->grid), cudaMemcpyHostToDevice);
cudaMemcpy(h_p_in.infectionDurationGrid, p->infectionDurationGrid, world_size * sizeof(*p->infectionDurationGrid), cudaMemcpyHostToDevice);

// Update `d_p_in` with the correct GPU addresses
cudaMemcpy(d_p_in, &h_p_in, sizeof(world_t), cudaMemcpyHostToDevice);

dim3 blockDim(16, 16);
dim3 gridDim((p->params.worldWidth + blockDim.x - 1) / blockDim.x, 
             (p->params.worldHeight + blockDim.y - 1) / blockDim.y);


world_update_cuda<<<gridDim, blockDim>>>(d_p_in, d_p_out, d_tmp_world);
cudaDeviceSynchronize();

cudaMemcpy(p->grid, h_p_in.grid, world_size * sizeof(*p->grid), cudaMemcpyDeviceToHost);

cudaFree(h_p_in.grid);
cudaFree(h_p_in.infectionDurationGrid);
cudaFree(d_p_in);
cudaFree(d_p_out);
cudaFree(d_tmp_world);

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
