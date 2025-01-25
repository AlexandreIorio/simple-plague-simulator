#ifdef __CUDACC__

#include <cuda_runtime.h>
#include <curand_kernel.h>
#define CUDA_SM		 128
#define CUDA_WARP_SIZE	 32
#define CUDA_BLOCK_DIM_X 16
#define CUDA_BLOCK_DIM_Y 16
#define CUDA_NB_THREAD	 (CUDA_BLOCK_DIM_X * CUDA_BLOCK_DIM_Y)
#define CUDA_NB_BLOCK	 (CUDA_SM / CUDA_WARP_SIZE)

typedef struct {
	world_t *d_world;
	state_t *d_curr_grid;
	// store this so we can free them later
	state_t *d_tmp_grid;
	uint8_t *d_infection_duration_grid;
} cuda_prepare_update_t;

static cuda_prepare_update_t cuda_prepare;

__global__ void setup_kernel(curandState *state, uint64_t seed)
{
	size_t i = blockIdx.y * blockDim.y + threadIdx.y;
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;

	int index = i * gridDim.x * blockDim.x + j;
	curand_init(seed, index, 0, &state[index]);
}
__global__ void generate_randoms(curandState *globalState, float *randoms)
{
	size_t i = blockIdx.y * blockDim.y + threadIdx.y;
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;

	int index = i * gridDim.x * blockDim.x + j;
	curandState localState = globalState[index];
	randoms[index] = curand_uniform(&localState);
	globalState[index] = localState;
}

static inline __device__ bool should_happen(int probability, curandState *state,
					    int threadIdX, int threadIdY)
{
	return probability < (curand(&state[threadIdX + threadIdY]) % 100);
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

bool __device__ world_should_infect(const world_t *p, curandState *state,
				    size_t i, size_t j, int probability)
{
	return world_get_nb_infected_neighbours(p, i, j) &&
	       cuda_should_happen(probability, state, i, j);
}
void __device__ world_infect_if_should_infect(const world_t *p, state_t *world,
					      curandState *state, size_t i,
					      size_t j, int probability)
{
	if (world_should_infect(p, state, i, j, probability)) {
		world[i * p->params.worldWidth + j] = INFECTED;
	}
}
void __device__ world_handle_infected(world_t *p, state_t *world,
				      curandState *state, size_t i, size_t j)
{
	const size_t index = i * p->params.worldWidth + j;

	if (p->infectionDurationGrid[index] == 0) {
		if (cuda_should_happen(p->params.deathProbability, state, i,
				       j)) {
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

static __global__ void world_update(world_t *d_p_in, state_t *d_tmp_world,
				    curandState *state)
{
	size_t i = blockIdx.y * blockDim.y + threadIdx.y;
	size_t j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < d_p_in->params.worldHeight && j < d_p_in->params.worldWidth) {
		int index = i * d_p_in->params.worldWidth + j;
		switch (d_p_in->grid[index]) {
		case HEALTHY:
			world_infect_if_should_infect(
				d_p_in, d_tmp_world, state, i, j,
				d_p_in->params.healthyInfectionProbability);
			break;
		case IMMUNE:
			world_infect_if_should_infect(
				d_p_in, d_tmp_world, state, i, j,
				d_p_in->params.immuneInfectionProbability);
			break;
		case INFECTED:
			world_handle_infected(d_p_in, d_tmp_world, state, i, j);
			break;
		case EMPTY:
		case DEAD:
			break;
		}
	}
	size_t index = i * d_p_in->params.worldWidth + j;
	d_p_in->grid[index] = d_tmp_world[index];
}
void world_update(world_t *p, void *raw)
{
	cuda_prepare_update_t *update_data = (cuda_prepare_update_t *)raw;

	dim3 blockDim(CUDA_BLOCK_DIM_X, CUDA_BLOCK_DIM_Y);
	dim3 gridDim((p->params.worldWidth + blockDim.x - 1) / blockDim.x,
		     (p->params.worldHeight + blockDim.y - 1) / blockDim.y);

	cuda_world_update<<<gridDim, blockDim> > >(d_p_in, d_tmp_world);

	cudaMempcy(p->grid, update_data->d_tmp_grid, cudaMemcpyDeviceToHost);
	cudaMempcy(p->infectionDurationGrid,
		   update_data->d_infection_duration_grid,
		   cudaMemcpyDeviceToHost);

	cudaFree(update_data->d_tmp_grid);
	cudaFree(update_data->d_grid);
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
	err = cudaMalloc((void **)&(d_tmp_world),
			 world_size * sizeof(*d_tmp_world));

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

	cudaMemcpy(d_tmp_grid, p->grid, world_size * sizeof(*d_tmp_grid));
	cudaMemcpy(d_grid, p->grid, world_size * sizeof(*d_grid));
	cudaMemcpy(d_infection_duration_grid, p->infectionDurationGrid,
		   world_size * sizeof(*d_infection_duration_grid));

	cudaMemcpy(d_world, &world, sizeof(world_t), cudaMemcpyHostToDevice);

	cuda_prepare.d_world = d_world;
	cuda_prepare.d_tmp_grid = d_tmp_grid;
	cuda_prepare.d_grid = d_grid;
	cuda_prepare.d_infection_duration_grid = d_infection_duration_grid;

	return (void *)&cuda_prepare;
}

#endif
