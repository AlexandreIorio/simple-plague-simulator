#include <stddef.h>
#include <stdint.h>

#ifdef __CUDACC__
#include <curand_kernel.h>
#include <curand.h>
#endif

#ifndef world_H
#define world_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { EMPTY = 0, HEALTHY, INFECTED, DEAD, IMMUNE } state_t;

typedef struct {
	size_t worldHeight;
	size_t worldWidth;
	size_t populationPercent;
	size_t initialInfected;
	size_t initialImmune;
	int deathProbability;
	int infectionDuration;
	int healthyInfectionProbability;
	int immuneInfectionProbability;
	int proximity;
} world_parameters_t;

typedef struct {
	world_parameters_t params;
	state_t *grid;
	uint8_t *infectionDurationGrid;
#ifdef __CUDACC__
	curandState *random_state;
#endif
} world_t;

///@brief inits the world passed with the params passed
///@param the world object to initialize
///@param the params to initialize world
///@return 0 on success -1 on error
int world_init(world_t *world, const world_parameters_t *p);

///@brief counts the number of people infected
///@param the world
///@return the number of people infected in the world
size_t world_get_infected(const world_t *w);

///@brief counts the number of healthy people.
///	 /!\ it doesn't count the immune people
///@param the world
///@return the number of healthy people in the world
size_t world_get_healthy(const world_t *w);

///@brief counts the number of immune people.
///	 /!\ it doesn't count the healthy people
///@param the world
///@return the number of immune people in the world
size_t world_get_immune(const world_t *w);

///@brief counts the number of dead people.
///@param the world
///@return the number of dead people in the world
size_t world_get_dead(const world_t *w);

///@brief prepare before running an update.
///	  Having this function ensures that world_update will never fail to run
///	and so its time can be measured more easily
///@param the world
///@return a pointer that must be checked before passing it to `world_update`
///	   If it's NULL a problem arose and you shouldn't call `world_update`

void *world_prepare_update(const world_t *p);

///@brief runs one round
///@param the world
///@param tmp: value returned by world_prepare_update.
///	  Can't be NULL
///@return 0 on success -1 on error
void world_update(world_t *w, void *tmp);

///@brief destroys world and everything inside it
///@param the world
void world_destroy(world_t *w);

#ifdef __cplusplus
}
#endif

#endif // world_H
