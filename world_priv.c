
#ifdef _OPENMP
#include <omp.h>
#endif
#include "world_priv.h"
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

size_t world_initial_population(const world_parameters_t *p)
{
	return world_world_size(p) * p->population_percentage / 100;
}

size_t world_world_size(const world_parameters_t *p)
{
	return p->width * p->height;
}

int world_init_common(world_t *world, const world_parameters_t *p)
{
	if (!world) {
		return -1;
	}

	const size_t world_size = world_world_size(p);
	if (!world_size) {
		return -1;
	}

	world->grid = (state_t *)malloc(world_size * sizeof(*world->grid));

	world->infection_duration_grid = (uint8_t *)malloc(
		world_size * sizeof(*world->infection_duration_grid));

	if (!world->grid || !world->infection_duration_grid) {
		return -1;
	}
	memset(world->grid, EMPTY, sizeof(*world->grid) * world_size);
	memset(world->infection_duration_grid, p->infection_duration,
	       sizeof(*world->infection_duration_grid) * world_size);
	memcpy(&world->params, p, sizeof(*p));

	return 0;
}

void world_destroy_common(world_t *w)
{
	free(w->grid);
	free(w->infection_duration_grid);
}
