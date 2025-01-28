#include <stdio.h>
#include <string.h>
#include "timeline.h"

const uint32_t FLAG = 0x1234cafe;

int timeline_init(timeline_t *tl, const world_parameters_t *params,
		  const char *path, size_t max_size)
{
	tl->fp = fopen(path, "wb");
	if (!tl->fp) {
		return -1;
	}

	tl->saved_rounds = 0;
	tl->max_size = max_size;
	memcpy(&tl->params, params, sizeof(tl->params));

	fwrite(&FLAG, sizeof(FLAG), 1, tl->fp);
	fwrite(&tl->saved_rounds, sizeof(tl->saved_rounds), 1, tl->fp);
	fwrite(&tl->params, sizeof(tl->params), 1, tl->fp);

	return 0;
}

size_t timeline_expected_size(const world_parameters_t *params,
			      size_t nb_rounds)
{
	return params->worldWidth * params->worldHeight * sizeof(uint8_t) *
		       nb_rounds + // Grid size * nb of rounds
	       sizeof(FLAG) +
	       sizeof(size_t);
}
timeline_error_t timeline_push_round(timeline_t *tl, uint8_t *grid)
{
	if (timeline_expected_size(&tl->params, tl->saved_rounds) >
	    tl->max_size) {
		return TL_MAX_SIZE;
	}
	fwrite(grid, sizeof(*grid),
	       tl->params.worldWidth * tl->params.worldHeight, tl->fp);
	++tl->saved_rounds;
	return TL_OK;
}

timeline_error_t timeline_save(timeline_t *tl)
{
	fseek(tl->fp, 0, SEEK_SET);
	fwrite(&FLAG, sizeof(FLAG), 1, tl->fp);
	fwrite(&tl->saved_rounds, sizeof(tl->saved_rounds), 1, tl->fp);
	fclose(tl->fp);
	return TL_OK;
}
