#ifndef WORLD_PRIV_H
#define WORLD_PRIV_H
#include "world.h"

#ifdef __cplusplus
extern "C" {
#endif

int world_init_common(world_t *world, const world_parameters_t *p);
size_t world_world_size(const world_parameters_t *p);
void world_destroy_common(world_t *w);

#ifdef __cplusplus
}
#endif

#endif
