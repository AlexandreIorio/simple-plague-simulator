
#include "world_priv.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t world_initial_population(const world_parameters_t *p)
{
	return world_world_size(p) * p->populationPercent / 100;
}

size_t world_world_size(const world_parameters_t *p)
{
	return p->worldWidth * p->worldHeight;
}

int world_init_common(world_t *world, const world_parameters_t *p)
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

#pragma omp parallel

	const size_t start_index = omp_get_thread_num() * chunk_per_thread;
	const bool is_last_thread = omp_get_thread_num() == max_threads - 1;
	const size_t people_to_spawn = is_last_thread ?
					       people_to_spawn_on_last_thread :
					       people_to_spawn_per_thread;
	const size_t infected_people_to_spawn =
		is_last_thread ? people_to_infect_on_last_thread :
				 people_to_infect_per_thread;
	const size_t immune_people_to_spawn =
		is_last_thread ? people_to_immunize_on_last_thread :
				 people_to_immunize_per_thread;
#else
	const size_t infected_people_to_spawn = p->initialInfected;
	const size_t immune_people_to_spawn = p->initialImmune;
#endif
	size_t people = 0;
	size_t people_infected = 0;
	size_t people_immune = 0;
	
	while (people < people_to_spawn) {
		const size_t i = start_index + (rand() % chunk_per_thread);
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
	return 0;
}

void world_destroy_common(world_t *w)
{
	free(w->grid);
	free(w->infectionDurationGrid);
}
