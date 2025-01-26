#include <stddef.h>
#include <stdint.h>


#ifndef WORLD_H
#define WORLD_H

#ifdef __cplusplus
extern "C" {
#endif

enum state { EMPTY = 0, HEALTHY, INFECTED, DEAD, IMMUNE };

typedef uint8_t state_t;

typedef struct {
	size_t worldHeight;
	size_t worldWidth;
	size_t populationPercent;
	size_t initialInfected;
	size_t initialImmune;
	int32_t deathProbability;
	int32_t infectionDuration;
	int32_t healthyInfectionProbability;
	int32_t immuneInfectionProbability;
	int32_t proximity;
} world_parameters_t;

typedef struct {
	state_t *grid;
	uint8_t *infectionDurationGrid;
	/* 
	* only used by cuda. Can't use a #ifdef __CUDACC__ as this file
	* is included in .c and .cpp files and when those files 
	* get compiled, CUDACC is not defined
	*/
	void *cuda_random_state; 
	world_parameters_t params;
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
