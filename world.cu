#ifdef __CUDACC__

#include <iostream>
#include <sstream>
#include "world_priv.h"
#include <cuda_runtime.h>
#define CUDA_SM 128
#define CUDA_WARP_SIZE 32
#define CUDA_BLOCK_DIM_X 16
#define CUDA_BLOCK_DIM_Y 16
#define CUDA_NB_THREAD (CUDA_BLOCK_DIM_X * CUDA_BLOCK_DIM_Y)
#define CUDA_NB_BLOCK (CUDA_SM / CUDA_WARP_SIZE)

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

__global__ void init_random_numbers(world_t *world) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;
	size_t j = blockIdx.x * blockDim.x + threadIdx.x; 
    const size_t index = i * world->params.worldWidth + j;
    const size_t world_size = world->params.worldWidth * world->params.worldHeight;
    if (index < world_size) {   
        curand_init(index, index, 0, &world->dStates[index]);
    }
}

__global__ void generate_random_numbers(world_t *world) {

	size_t i = blockIdx.y * blockDim.y + threadIdx.y;
	size_t j = blockIdx.x * blockDim.x + threadIdx.x; 
    const size_t index = i * world->params.worldWidth + j;
    const size_t world_size = world->params.worldWidth * world->params.worldHeight;

    if (index < world_size) {   
        world->dRandom[index] = curand_uniform(&world->dStates[index]);
    }
}

static inline __device__ bool should_happen(int probability, float random_float)
{
    return random_float < ((double)probability / 100);
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
    curandState_t *d_states;
    float *d_random;
    const size_t world_size = world_world_size(p);

	int err = world_init_common(world, p);
	if (err < 0) {
		return err;
	}

    checkCudaErrors(cudaMalloc((void **)&d_states, world_size * sizeof(*d_states)));
    checkCudaErrors(cudaMalloc((void **)&d_random, world_size * sizeof(*d_random)));

    world->dStates = d_states;
    world->dRandom = d_random;

	dim3 block(CUDA_BLOCK_DIM_X, CUDA_BLOCK_DIM_Y);
	dim3 grid((world->params.worldWidth + block.x - 1) / block.x,
		  (world->params.worldHeight + block.y - 1) / block.y);
 
    init_random_numbers<<<grid, block>>>(world);

	return 0;
}

bool __device__ world_should_infect(world_t *p, size_t i, size_t j,
				    int probability)
{
	return world_get_nb_infected_neighbours(p, i, j) &&
	       should_happen(probability, p->dRandom[i * p->params.worldWidth + j]);
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
		if (should_happen(p->params.deathProbability, p->dRandom[index])) {
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
		w->grid[index] = result_grid[index];
	}
}
void world_update(world_t *p, void *raw)
{
	cuda_prepare_update_t *update_data = (cuda_prepare_update_t *)raw;
	const size_t world_size = world_world_size(&p->params);
	const size_t GRID_SIZE = world_size * sizeof(state_t);
	const size_t INFECTION_GRID_SIZE = world_size * sizeof(uint8_t);

	dim3 block(CUDA_BLOCK_DIM_X, CUDA_BLOCK_DIM_Y);
	dim3 grid((p->params.worldWidth + block.x - 1) / block.x,
		  (p->params.worldHeight + block.y - 1) / block.y);

    generate_random_numbers<<<grid, block>>>(update_data->d_world);

    checkCudaErrors(cudaDeviceSynchronize()); 

	world_update_k<<<grid, block>>>(update_data->d_world,
					  update_data->d_tmp_grid);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }

	checkCudaErrors(cudaMemcpy(p->grid, update_data->d_tmp_grid, GRID_SIZE,
				   cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(p->infectionDurationGrid,
				   update_data->d_infection_duration_grid,
				   INFECTION_GRID_SIZE,
				   cudaMemcpyDeviceToHost));

	cudaFree(update_data->d_tmp_grid);
	cudaFree(update_data->d_curr_grid);
	cudaFree(update_data->d_infection_duration_grid);
	cudaFree(update_data->d_world);
}
void *world_prepare_update(const world_t *p)
{
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

	if (!d_grid) {
		FatalError("d_grid is null");
	}
	if (!d_tmp_grid) {
		FatalError("d_tmp_grid is null");
	}
	if (!d_infection_duration_grid) {
		FatalError("d_infection_duration_grid is null");
	}

	if (!p || !p->grid) {
		FatalError("p || p->grid is null");
	}
	world_t world;

	world.grid = d_grid;
	world.infectionDurationGrid = d_infection_duration_grid;
	world.params = p->params;

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

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    }

	return (void *)&cuda_prepare;
}

void world_destroy(world_t *w)
{
	world_destroy_common(w);
}

#endif
