#ifdef __CUDACC__

#include "world_priv.h"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#define CUDA_SM		 128
#define CUDA_WARP_SIZE	 32
#define CUDA_BLOCK_DIM_X 16
#define CUDA_BLOCK_DIM_Y 16
#define CUDA_NB_THREAD	 (CUDA_BLOCK_DIM_X * CUDA_BLOCK_DIM_Y)
#define CUDA_NB_BLOCK	 (CUDA_SM / CUDA_WARP_SIZE)

static __global__ void world_init_random_generator(curandState *state,
						uint64_t seed)
{
	const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
	const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
	const size_t index = i * gridDim.x * blockDim.x + j;
	curand_init(seed, index, 0, &state[index]);
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

typedef struct {
	world_t *d_world;
	state_t *d_curr_grid;
	// store this so we can free them later
	state_t *d_tmp_grid;
	uint8_t *d_infection_duration_grid;
} cuda_prepare_update_t;

static cuda_prepare_update_t cuda_prepare;

int world_init(world_t *world, const world_parameters_t *p)
{
	int err = world_init_common(world, p);
	if (err < 0) {
		return err;
	}
	curandState *d_state;
	cudaMalloc((void **)&d_state,
		   CUDA_NB_THREAD * CUDA_NB_BLOCK * sizeof(*d_state));
	dim3 block(CUDA_BLOCK_DIM_X, CUDA_BLOCK_DIM_Y);
	dim3 grid((p->worldWidth + block.x - 1) / block.x,
		  (p->worldHeight + block.y - 1) / block.y);

	world_init_random_generator<<<grid, block>>>(d_state, 1337);

	/* No need to synchronize here */
	world->random_state = d_state;

	return 0;
}

bool __device__ world_should_infect(world_t *p, size_t i, size_t j,
				    int probability)
{
	return world_get_nb_infected_neighbours(p, i, j) &&
	       should_happen(probability,
			     &p->random_state[i * p->params.worldWidth + j]);
}
void __device__ world_infect_if_should_infect(world_t *p, state_t *grid,
					      size_t i, size_t j,
					      int probability)
{
	if (world_should_infect(p, i, j, probability)) {
		grid[i * p->params.worldWidth + j] = INFECTED;
	}
}
void __device__ world_handle_infected(world_t *p, state_t *world, size_t i,
				      size_t j)
{
	const size_t index = i * p->params.worldWidth + j;

	if (p->infectionDurationGrid[index] == 0) {
		if (should_happen(p->params.deathProbability,
				  &p->random_state[index])) {
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
		int index = i * w->params.worldWidth + j;
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
	size_t index = i * w->params.worldWidth + j;
	w->grid[index] = result_grid[index];
}
void world_update(world_t *p, void *raw)
{
	cuda_prepare_update_t *update_data = (cuda_prepare_update_t *)raw;
	const size_t world_size = world_world_size(&p->params);

	dim3 block(CUDA_BLOCK_DIM_X, CUDA_BLOCK_DIM_Y);
	dim3 grid((p->params.worldWidth + blockDim.x - 1) / blockDim.x,
		  (p->params.worldHeight + blockDim.y - 1) / blockDim.y);
	world_update_k<<<grid, block>>>(update_data->d_world,
					  update_data->d_tmp_grid);

	cudaMemcpy(p->grid, update_data->d_tmp_grid,
		   world_size * sizeof(*p->grid), cudaMemcpyDeviceToHost);
	cudaMemcpy(p->infectionDurationGrid,
		   update_data->d_infection_duration_grid,
		   world_size * sizeof(*p->infectionDurationGrid),
		   cudaMemcpyDeviceToHost);

	cudaFree(update_data->d_tmp_grid);
	cudaFree(update_data->d_curr_grid);
	cudaFree(update_data->d_infection_duration_grid);
	cudaFree(update_data->d_world);
}
void *world_prepare_update(const world_t *p)
{
	const size_t world_size = world_world_size(&p->params);
	int err;
	state_t *d_grid;
	err = cudaMalloc((void **)&(d_grid), world_size * sizeof(*d_grid));
	if (err != cudaSuccess) {
		return NULL;
	}
	state_t *d_tmp_grid;
	err = cudaMalloc((void **)&(d_tmp_grid),
			 world_size * sizeof(*d_tmp_grid));

	if (err != cudaSuccess) {
		cudaFree(d_grid);
		return NULL;
	}

	uint8_t *d_infection_duration_grid;

	err = cudaMalloc((void **)&(d_infection_duration_grid),
			 world_size * sizeof(*d_infection_duration_grid));

	if (err != cudaSuccess) {
		cudaFree(d_grid);
		cudaFree(d_tmp_grid);
		return NULL;
	}
	world_t world;

	world.grid = d_grid;
	world.infectionDurationGrid = d_infection_duration_grid;
	world.params = p->params;

	world_t *d_world;

	err = cudaMalloc((void **)&(d_world), sizeof(*d_world));

	if (err != cudaSuccess) {
		cudaFree(d_grid);
		cudaFree(d_tmp_grid);
		cudaFree(d_infection_duration_grid);
		return NULL;
	}

	cudaMemcpy(d_tmp_grid, p->grid, world_size * sizeof(*d_tmp_grid),
		   cudaMemcpyHostToDevice);
	cudaMemcpy(d_grid, p->grid, world_size * sizeof(*d_grid),
		   cudaMemcpyHostToDevice);
	cudaMemcpy(d_infection_duration_grid, p->infectionDurationGrid,
		   world_size * sizeof(*d_infection_duration_grid),
		   cudaMemcpyHostToDevice);

	cudaMemcpy(d_world, &world, sizeof(world_t), cudaMemcpyHostToDevice);

	cuda_prepare.d_world = d_world;
	cuda_prepare.d_curr_grid = d_grid;
	cuda_prepare.d_tmp_grid = d_tmp_grid;
	cuda_prepare.d_infection_duration_grid = d_infection_duration_grid;

	return (void *)&cuda_prepare;
}

void world_destroy(world_t *w)
{
	cudaFree(w->random_state);
	world_destroy_common(w);
}

#endif
