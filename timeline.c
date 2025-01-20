#include <stdio.h>
#include <string.h>
#include "timeline.h"

const uint32_t FLAG = 0x1234cafe;

int timeline_init(timeline_t *tl, const world_parameters_t *params,
		  const char *path)
{
	tl->fp = fopen(path, "wb");
	if (!tl->fp) {
		return -1;
	}

	tl->nb_rounds = 0;
	memcpy(&tl->params, params, sizeof(tl->params));

	fwrite(&FLAG, sizeof(FLAG), 1, tl->fp);
	fwrite(&tl->nb_rounds, sizeof(tl->nb_rounds), 1, tl->fp);
	fwrite(&tl->params, sizeof(tl->params), 1, tl->fp);

	return 0;
}

int timeline_push_round(timeline_t *tl, int *grid)
{
	// fwrite(grid, sizeof(*grid),
	//        tl->params.worldWidth * tl->params.worldHeight, tl->fp);

	++tl->nb_rounds;
	return 0;
}

int timeline_save(timeline_t *tl)
{
	fseek(tl->fp, 0, SEEK_SET);
	fwrite(&FLAG, sizeof(FLAG), 1, tl->fp);
	fwrite(&tl->nb_rounds, sizeof(tl->nb_rounds), 1, tl->fp);
	fclose(tl->fp);
	return 0;
}
