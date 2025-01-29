#include "world.h"
#include "world_priv.h"

#include <stdio.h>

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
void world_print_params(const world_parameters_t *params)
{
	printf("Population Percentage          %d%%\n",
	       params->population_percentage);
	printf("World Width                    %zu\n", params->width);
	printf("World Height                   %zu\n", params->height);
	printf("Proximity                      %zu\n", params->proximity);
	printf("Infection Duration             %d rounds\n",
	       params->infection_duration);
	printf("Healthy Infection Probability  %d%%\n",
	       params->healthy_infection_probability);
	printf("Immune Infection Probability   %d%%\n",
	       params->immune_infection_probability);
	printf("Death Probability              %d%%\n",
	       params->death_probability);
	printf("Initial Infected               %zu\n",
	       params->initial_infected);
	printf("Initial Immune                 %zu\n", params->initial_immune);
}
