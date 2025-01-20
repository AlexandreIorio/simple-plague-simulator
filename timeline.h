#include <stddef.h>
#include <stdio.h>
#include "world.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
	FILE *fp;
	size_t nb_rounds;
	world_parameters_t params;
} timeline_t;

int timeline_init(timeline_t *tl, const world_parameters_t *params,
		  const char *path);

int timeline_push_round(timeline_t *tl, int *grid);

int timeline_save(timeline_t *tl);

#ifdef __cplusplus
}
#endif
