#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "timeline.h"

#define DEFAULT_ROUNDS 128
const uint32_t FLAG = 0x1234cafe;

int timeline_init(timeline_t *tl, const world_parameters_t *params)
{
	tl->grids = (int **)calloc(DEFAULT_ROUNDS, sizeof(*tl->grids));
	if (!tl->grids) {
		return -1;
	}
	tl->grids_size = DEFAULT_ROUNDS;
	tl->nb_rounds = 0;

	memcpy(&tl->params, params, sizeof(tl->params));
	return 0;
}

int timeline_push_round(timeline_t *tl, int *grid)
{
	if (tl->nb_rounds >= tl->grids_size) {
		const size_t new_size = tl->grids_size * 2;
		int **new_grids = (int **)realloc(
			tl->grids, tl->grids_size * tl->params.worldWidth *
					   tl->params.worldHeight *
					   sizeof(**tl->grids));
		if (!new_grids) {
			return -1;
		}
		tl->grids_size = new_size;
		tl->grids = new_grids;
	}

	tl->grids[tl->nb_rounds] =
		(int *)malloc(tl->params.worldWidth * tl->params.worldHeight *
			      sizeof(**tl->grids));

	if (!tl->grids) {
		return -1;
	}

	memcpy(tl->grids[tl->nb_rounds], grid,
	       tl->params.worldWidth * tl->params.worldHeight *
		       sizeof(*tl->grids[tl->nb_rounds]));
	++tl->nb_rounds;
	return 0;
}

int timeline_save(timeline_t *tl, const char *path)
{
	FILE *f = fopen(path, "wb");

	if (!f) {
		return -1;
	}
	fwrite(&FLAG, sizeof(FLAG), 1, f);
	fwrite(&tl->nb_rounds, sizeof(tl->nb_rounds), 1, f);
	fwrite(&tl->params, sizeof(tl->params), 1, f);

	for (size_t i = 0; i < tl->nb_rounds; ++i) {
		printf("Saving round %zu/%zu, size (%zu) * (%zu)\n", i,
		       tl->nb_rounds, sizeof(**tl->grids),
		       tl->params.worldWidth * tl->params.worldHeight);
		fwrite(tl->grids[i], sizeof(**tl->grids),
		       tl->params.worldWidth * tl->params.worldHeight, f);
	}
	fclose(f);
	return 0;
}

int timeline_read_from_file(timeline_t *tl, const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f) {
		return -1;
	}
	uint32_t flag;
	world_parameters_t params;
	uint32_t nb_rounds;
	fread(&flag, sizeof(flag), 1, f);
	if (flag != FLAG) {
		return -1;
	}

	fread(&nb_rounds, sizeof(tl->nb_rounds), 1, f);
	fread(&params, sizeof(params), 1, f);

	tl->grids = (int **)calloc(nb_rounds, sizeof(*tl->grids));

	if (!tl->grids) {
		fclose(f);
		return -1;
	}

	memcpy(&tl->params, &params, sizeof(tl->params));

	for (size_t i = 0; i < tl->nb_rounds; ++i) {
		tl->grids[i] = (int *)malloc(tl->params.worldWidth *
					     tl->params.worldHeight *
					     sizeof(**tl->grids));

		if (!tl->grids[i]) {
			for (size_t j = 0; j < i; ++j) {
				free(tl->grids[j]);
			}
			free(tl->grids);
			fclose(f);
			return -1;
		}

		fread(tl->grids[i], sizeof(**tl->grids),
		      tl->params.worldWidth * tl->params.worldHeight, f);
	}
	fclose(f);
	return 0;
}
