#include "world.h"
#include "world_priv.h"

static size_t get_in_state(const world_t *p, state_t state)
{
	const size_t size = world_world_size(&p->params);
	size_t sum = 0;
	for (size_t i = 0; i < size; ++i) {
		sum += p->grid[i] == state;
	}
	return sum;
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
