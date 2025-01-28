#ifdef __CUDACC__

#include <iostream>
#include <sstream>
#include "world_priv.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#define CUDA_SM		 128
#define CUDA_WARP_SIZE	 32
#define CUDA_BLOCK_DIM_X 32
#define CUDA_BLOCK_DIM_Y 32
#define CUDA_NB_THREAD	 (CUDA_BLOCK_DIM_X * CUDA_BLOCK_DIM_Y)
#define CUDA_NB_BLOCK	 (CUDA_SM / CUDA_WARP_SIZE)

#define FatalError(s)                                                          \
	do {                                                                   \
		std::cout << std::flush << "ERROR: " << s << " in "            \
			  << __FILE__ << ':' << __LINE__ << "\nAborting...\n"; \
		cudaDeviceReset();                                             \
		exit(-1);                                                      \
	} while (0)

#define checkCudaErrors(status)                                                \
	do {                                                                   \
		std::stringstream _err;                                        \
		if (status != 0) {                                             \
			_err << "cuda failure (" << cudaGetErrorString(status) \
			     << ')';                                           \
			FatalError(_err.str());                                \
		}                                                              \
	} while (0)

static __global__ void world_init_random_generator(curandState *state,
						   size_t len, uint64_t seed)
{
	const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t index = i * gridDim.x * blockDim.x + j;
	if (index < len) {
		curand_init(seed, index, 0, &state[index]);
	}
}

static inline __device__ bool should_happen(int probability, curandState *state)
{
	double rand_value = curand_uniform(state);
	return rand_value < ((double)probability / 100);
}

static __device__ uint8_t world_get_nb_infected_neighbours(const world_t *p,
							   size_t i, size_t j)
{
	uint8_t sum = 0;
	for (int dx = -p->params.proximity; dx <= p->params.proximity; ++dx) {
		for (int dy = -p->params.proximity; dy <= p->params.proximity;
		     ++dy) {
			if (dx == 0 && dy == 0) {
				continue;
			}
			const int ni = i + dx;
			const int nj = j + dy;
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

typedef struct {
	world_t *d_world;
	state_t *d_curr_grid;
	// store this so we can free them later
	state_t *d_tmp_grid;
	uint8_t *d_infection_duration_grid;
} cuda_prepare_update_t;

static cuda_prepare_update_t cuda_prepare;

static __global__ void init_population_kernel(
	state_t *grid, const world_parameters_t *p, size_t people_to_spawn,
	curandState *random_states,
	int *occupation_buffer) // Buffer used to lock a random position
{
	const size_t world_size = p->worldWidth * p->worldHeight;
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= p->worldHeight || j >= p->worldWidth) {
		return;
	}

	int index = i * p->worldWidth + j;
	if (index >= people_to_spawn) {
		return;
	}

	state_t state;
	if (index < p->initialInfected) {
		state = INFECTED;
	} else if (index < p->initialInfected + p->initialImmune) {
		state = IMMUNE;
	} else {
		state = HEALTHY;
	}

	bool found_position = false;
	while (!found_position) {
		size_t pos = curand(&random_states[index]) % world_size;
		// if occupation == 0 then write 1 to define cell usage
		if (atomicCAS(&occupation_buffer[pos], 0, 1) == 0) {
			grid[pos] = state;
			found_position = true;
		}
	}
}

int world_init(world_t *world, const world_parameters_t *p)
{
	int err = world_init_common(world, p);
	if (err < 0) {
		return err;
	}

	curandState *d_state;
	const size_t world_size = world_world_size(p);
	dim3 block(CUDA_BLOCK_DIM_X, CUDA_BLOCK_DIM_Y);
	dim3 grid((p->worldWidth + CUDA_BLOCK_DIM_X - 1) / CUDA_BLOCK_DIM_X,
		  (p->worldHeight + CUDA_BLOCK_DIM_Y - 1) / CUDA_BLOCK_DIM_Y);
	checkCudaErrors(
		cudaMalloc((void **)&d_state, world_size * sizeof(*d_state)));

	world_init_random_generator<<<grid, block>>>(d_state, world_size,
						       1337);
	checkCudaErrors(cudaDeviceSynchronize());

	world->cuda_random_state = (void *)d_state;

	state_t *d_grid;

	cudaMalloc(&d_grid, world_size * sizeof(state_t));

	cudaMemset(d_grid, EMPTY, world_size * sizeof(state_t));

	int *d_occupation_buffer;
	cudaMalloc((void **)&d_occupation_buffer,
		   world_size * sizeof(*d_occupation_buffer));

	cudaMemset(d_occupation_buffer, 0,
		   world_size * sizeof(*d_occupation_buffer));
	world_parameters_t *d_p;

	cudaMalloc((void **)&d_p, sizeof(*d_p));
	cudaMemcpy(d_p, p, sizeof(*d_p), cudaMemcpyHostToDevice);

	const size_t people_to_spawn = world_initial_population(p);

	init_population_kernel<<<grid, block>>>(d_grid, d_p, people_to_spawn,
						  d_state, d_occupation_buffer);

	checkCudaErrors(cudaDeviceSynchronize());

	cudaMemcpy(world->grid, d_grid, world_size * sizeof(state_t),
		   cudaMemcpyDeviceToHost);

	cudaFree(d_grid);

	return 0;
}

bool __device__ world_should_infect(world_t *p, size_t i, size_t j,
				    int probability)
{
	return world_get_nb_infected_neighbours(p, i, j) &&
	       should_happen(probability,
			     &((curandState *)p->cuda_random_state)
				     [i * p->params.worldWidth + j]);
}
void __device__ world_infect_if_should_infect(world_t *p, state_t *grid,
					      size_t i, size_t j,
					      int probability)
{
	if (world_should_infect(p, i, j, probability)) {
		const size_t index = i * p->params.worldWidth + j;
		grid[index] = INFECTED;
		p->infectionDurationGrid[index] = p->params.infectionDuration;
	}
}
void __device__ world_handle_infected(world_t *p, state_t *world, size_t i,
				      size_t j)
{
	const size_t index = i * p->params.worldWidth + j;

	if (p->infectionDurationGrid[index] == 0) {
		if (should_happen(
			    p->params.deathProbability,
			    &((curandState *)p->cuda_random_state)[index])) {
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

static __global__ void world_update_k(world_t *w, state_t *result_grid)
{
	size_t i = blockIdx.y * blockDim.y + threadIdx.y;
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < w->params.worldHeight && j < w->params.worldWidth) {
		size_t index = i * w->params.worldWidth + j;

		result_grid[index] = w->grid[index];

		switch (w->grid[index]) {
		case HEALTHY:
			world_infect_if_should_infect(
				w, result_grid, i, j,
				w->params.healthyInfectionProbability);
			break;
		case IMMUNE:
			world_infect_if_should_infect(
				w, result_grid, i, j,
				w->params.immuneInfectionProbability);
			break;
		case INFECTED:
			world_handle_infected(w, result_grid, i, j);
			break;
		case EMPTY:
		case DEAD:
			break;
		}
	}
}
void world_update(world_t *p, void *raw)
{
	(void)raw;
	const size_t world_size = world_world_size(&p->params);
	const size_t GRID_SIZE = world_size * sizeof(state_t);
	const size_t INFECTION_GRID_SIZE = world_size * sizeof(uint8_t);
	state_t *d_grid;
	state_t *d_tmp_grid;
	uint8_t *d_infection_duration_grid;

	checkCudaErrors(cudaMalloc((void **)&d_grid, GRID_SIZE));
	checkCudaErrors(cudaMalloc((void **)&d_tmp_grid, GRID_SIZE));
	checkCudaErrors(cudaMalloc((void **)&d_infection_duration_grid,
				   INFECTION_GRID_SIZE));

	world_t world;

	world.grid = d_grid;
	world.infectionDurationGrid = d_infection_duration_grid;
	world.params = p->params;
	world.cuda_random_state = p->cuda_random_state;

	world_t *d_world;

	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(world_t)));

	checkCudaErrors(
		cudaMemcpy(d_grid, p->grid, GRID_SIZE, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_tmp_grid, p->grid, GRID_SIZE,
				   cudaMemcpyHostToDevice));
	checkCudaErrors(
		cudaMemcpy(d_infection_duration_grid, p->infectionDurationGrid,
			   INFECTION_GRID_SIZE, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaMemcpy(d_world, &world, sizeof(world_t),
				   cudaMemcpyHostToDevice));

	cuda_prepare.d_world = d_world;
	cuda_prepare.d_curr_grid = d_grid;
	cuda_prepare.d_tmp_grid = d_tmp_grid;
	cuda_prepare.d_infection_duration_grid = d_infection_duration_grid;

	size_t infected_before = world_get_infected(p);

	dim3 block(CUDA_BLOCK_DIM_X, CUDA_BLOCK_DIM_Y);
	dim3 grid((p->params.worldWidth + block.x - 1) / block.x,
		  (p->params.worldHeight + block.y - 1) / block.y);
	world_update_k<<<grid, block>>>(cuda_prepare.d_world,
					  cuda_prepare.d_tmp_grid);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(p->grid, cuda_prepare.d_tmp_grid, GRID_SIZE,
				   cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(p->infectionDurationGrid,
				   cuda_prepare.d_infection_duration_grid,
				   INFECTION_GRID_SIZE,
				   cudaMemcpyDeviceToHost));

	size_t infected_after = world_get_infected(p);

	cudaFree(cuda_prepare.d_tmp_grid);
	cudaFree(cuda_prepare.d_curr_grid);
	cudaFree(cuda_prepare.d_infection_duration_grid);
	cudaFree(cuda_prepare.d_world);
}
void *world_prepare_update(const world_t *p)
{
	return (void *)&cuda_prepare;
}

void world_destroy(world_t *w)
{
	cudaFree(w->cuda_random_state);
	world_destroy_common(w);
}

#endif
