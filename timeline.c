#include <stdio.h>
#include <string.h>
#include "timeline.h"

const uint16_t FLAG = 0xcafe;

timeline_error_t timeline_init(timeline_t *tl, const world_parameters_t *params,
			       const char *path, size_t max_size)
{
	tl->fp = fopen(path, "wb");
	if (!tl->fp) {
		return TL_FAILED_TO_OPEN_FILE;
	}

	tl->saved_rounds = 0;
	tl->max_size = max_size;
	tl->file_size = 0;
	memcpy(&tl->params, params, sizeof(tl->params));

	fwrite(&FLAG, sizeof(FLAG), 1, tl->fp);
	fwrite(&tl->saved_rounds, sizeof(tl->saved_rounds), 1, tl->fp);
	fwrite(&tl->params, sizeof(tl->params), 1, tl->fp);

	return TL_OK;
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
	for (size_t i = 1; i < grid_size; ++i) {
		if (count == 255 || grid[i] != element) {
			fwrite(&count, sizeof(count), 1, tl->fp);
			fwrite(&element, sizeof(element), 1, tl->fp);
			tl->file_size += 2;
			count = 1;
			element = grid[i];
		} else {
			count++;
		}
	}
	fwrite(&count, sizeof(count), 1, tl->fp);
	fwrite(&element, sizeof(element), 1, tl->fp);
	tl->file_size += 2;
	++tl->saved_rounds;
	return TL_OK;
}

timeline_error_t timeline_load(timeline_t *tl, const char *path)
{
	tl->fp = fopen(path, "rb");
	if (!tl->fp) {
		return TL_FAILED_TO_OPEN_FILE;
	}
	uint16_t flag;
	fread(&flag, sizeof(flag), 1, tl->fp);
	if (flag != FLAG) {
		fclose(tl->fp);
		return TL_INVALID_TIMELINE;
	}
	fread(&tl->saved_rounds, sizeof(tl->saved_rounds), 1, tl->fp);
	fread(&tl->params, sizeof(tl->params), 1, tl->fp);
	return TL_OK;
}

timeline_error_t timeline_get_round(timeline_t *tl, uint8_t *grid)
{
	const size_t grid_size = tl->params.worldWidth * tl->params.worldHeight;
	size_t i = 0;
	while (i < grid_size) {
		uint8_t count;
		uint8_t element;
		size_t read = fread(&count, sizeof(count), 1, tl->fp);
		if (read != 1) {
			return TL_END;
		}
		read = fread(&element, sizeof(element), 1, tl->fp);
		if (read != 1) {
			return TL_END;
		}
		for (uint8_t j = 0; j < count; ++j) {
			if (i >= grid_size) {
				return TL_INVALID_TIMELINE;
			}
			grid[i++] = element;
		}
	}
	return TL_OK;
}

timeline_error_t timeline_save(timeline_t *tl)
{
	fseek(tl->fp, 0, SEEK_SET);
	fwrite(&FLAG, sizeof(FLAG), 1, tl->fp);
	fwrite(&tl->saved_rounds, sizeof(tl->saved_rounds), 1, tl->fp);
	timeline_close(tl);
	return TL_OK;
}

timeline_error_t timeline_close(timeline_t *tl)
{
	fclose(tl->fp);
	return TL_OK;
}
