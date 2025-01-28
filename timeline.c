#include <stdio.h>
#include <string.h>
#include "timeline.h"

const uint16_t FLAG = 0xcafe;

int timeline_init(timeline_t *tl, const world_parameters_t *params,
		  const char *path, size_t max_size)
{
	tl->fp = fopen(path, "wb");
	if (!tl->fp) {
		return -1;
	}

	tl->saved_rounds = 0;
	tl->max_size = max_size;
	tl->file_size = 0;
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
	if (tl->file_size >= tl->max_size) {
		return TL_MAX_SIZE;
	}
	const size_t grid_size = tl->params.worldWidth * tl->params.worldHeight;
	uint8_t count = 1;
	uint8_t element = grid[0];
	for (size_t i = 0; i < grid_size; ++i) {
		if (count == 255 || grid[i] != element) {
			fwrite(&count, sizeof(count), 1, tl->fp);
			fwrite(&element, sizeof(*grid), 1, tl->fp);
			tl->file_size += 2;
			count = 1;
			element = grid[i];
		} else {
			count++;
		}
	}
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
