
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
	printf("Called World Size %zu %zu\n", p->worldWidth, p->worldHeight);
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

void world_destroy_common(world_t *w)
{
	free(w->grid);
	free(w->infectionDurationGrid);
}
