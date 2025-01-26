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
	return world_init_common(world, p);
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
	memcpy(p->grid, tmp_world, world_size * sizeof(*p->grid));
}

void world_destroy(world_t *w)
{
	world_destroy_common(w);
}
