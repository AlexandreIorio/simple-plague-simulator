#include <stddef.h>
#include "world.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
	int **grids;
	size_t nb_rounds;
	size_t grids_size;
	world_parameters_t params;
} timeline_t;

int timeline_init(timeline_t *tl, const world_parameters_t *params);

int timeline_push_round(timeline_t *tl, int *grid);

int timeline_save(timeline_t *tl, const char *path);

int timeline_read_from_file(timeline_t *tl, const char *path);

#ifdef __cplusplus
}
#endif
