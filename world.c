#include <assert.h>
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "world.h"
#include "world_priv.h"

static inline bool should_happen(int probability)
{
	return probability < (rand() % 100);
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

int world_init(world_t *world, const world_parameters_t *p)
{
	int ret = world_init_common(world, p);

    if (ret) {
        return -1;
    }

	const size_t world_size = world_world_size(p);
	if (!world_size) {
		return -1;
	}

    const size_t people_to_spawn = world_size * p->populationPercent / 100;


	if (!(people_to_spawn >= p->initialImmune + p->initialInfected)) {
		return -1;
	}

	world->grid = (state_t *)malloc(world_size * sizeof(*world->grid));

	world->infectionDurationGrid = (uint8_t *)malloc(
		world_size * sizeof(*world->infectionDurationGrid));

	if (!world->grid || !world->infectionDurationGrid) {
		return -1;
	}
	memset(world->grid, EMPTY, sizeof(*world->grid) * world_size);
	memset(world->infectionDurationGrid, p->infectionDuration,
	       sizeof(*world->infectionDurationGrid) * world_size);
	memcpy(&world->params, p, sizeof(*p));

	srand(time(NULL));

#ifdef _OPENMP
	const size_t max_threads = omp_get_max_threads();
	const size_t people_to_infect_per_thread =
		p->initialInfected / max_threads;
	const size_t people_to_infect_on_last_thread =
		p->initialInfected % max_threads;
	const size_t people_to_immunize_per_thread =
		p->initialImmune / max_threads;
	const size_t people_to_immunize_on_last_thread =
		p->initialImmune % max_threads;
	const size_t people_to_spawn_per_thread = people_to_spawn / max_threads;
	const size_t people_to_spawn_on_last_thread =
		people_to_spawn % max_threads;
	const size_t chunk_per_thread = world_size / max_threads;
	const size_t last_thread_chunk = world_size % max_threads;

#pragma omp parallel
	{
		const size_t start_index =
			omp_get_thread_num() * chunk_per_thread;
		const bool is_last_thread = omp_get_thread_num() ==
					    max_threads - 1;
		const size_t people_to_spawn =
			is_last_thread ?
				people_to_spawn_per_thread +
					people_to_spawn_on_last_thread :
				people_to_spawn_per_thread;
		const size_t infected_people_to_spawn =
			is_last_thread ?
				people_to_infect_per_thread +
					people_to_infect_on_last_thread :
				people_to_infect_per_thread;
		const size_t immune_people_to_spawn =
			is_last_thread ?
				people_to_immunize_per_thread +
					people_to_immunize_on_last_thread :
				people_to_immunize_per_thread;
		const size_t thread_chunk =
			is_last_thread ? last_thread_chunk + chunk_per_thread :
					 chunk_per_thread;
#else
	const size_t infected_people_to_spawn = p->initialInfected;
	const size_t immune_people_to_spawn = p->initialImmune;
#endif
		size_t people = 0;
		size_t people_infected = 0;
		size_t people_immune = 0;

		while (people < people_to_spawn) {
#ifdef _OPENMP
			const size_t i = start_index + (rand() % thread_chunk);
#else
		const size_t i = rand() % world_size;

#endif
			if (world->grid[i] != EMPTY) {
				continue;
			}
			++people;
			if (people_infected < infected_people_to_spawn) {
				world->grid[i] = INFECTED;
				++people_infected;
				continue;
			}
			if (people_immune < immune_people_to_spawn) {
				world->grid[i] = IMMUNE;
				++people_immune;
				continue;
			}
			world->grid[i] = HEALTHY;
		}
#ifdef _OPENMP
	}
#endif
    return 0;

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

void world_update(world_t *p, void *raw)
{
	const size_t world_size = world_world_size(&p->params);
	state_t *tmp_world = (state_t *)raw;
	memcpy(tmp_world, p->grid, world_size * sizeof(*p->grid));

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (size_t i = 0; i < p->params.worldHeight; i++) {
#ifdef _OPENMP
#pragma omp parallel for
#endif
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
	memcpy(p->grid, tmp_world, world_size * sizeof(*p->grid));
}

void world_destroy(world_t *w)
{
	world_destroy_common(w);
}
