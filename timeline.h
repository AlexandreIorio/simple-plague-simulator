#include <stddef.h>
#include <stdio.h>
#include "world.h"

#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
	FILE *fp;
	size_t saved_rounds;
	world_parameters_t params;
	size_t max_size;
	size_t file_size;
} timeline_t;

typedef enum {
	TL_FAILED_TO_OPEN_FILE = -1,
	TL_OK = 0,
	TL_MAX_SIZE = 1,
} timeline_error_t;

timeline_error_t timeline_init(timeline_t *tl, const world_parameters_t *params,
			       const char *path, size_t max_size);

timeline_error_t timeline_push_round(timeline_t *tl, uint8_t *grid);

timeline_error_t timeline_save(timeline_t *tl);

size_t timeline_expected_size(const world_parameters_t *params,
			      size_t nb_rounds);
#ifdef __cplusplus
}
#endif
