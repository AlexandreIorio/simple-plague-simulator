#include <stdint.h>
#include "world.h"
#include "timeline.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

static const char *CSV_HEADER = "round,healthy,infected,immune,dead,total\n";

int main(int argc, char *argv[])
{
	if (argc < 3) {
		printf("Usage %s <timeline_file.bin> <output.csv>\n", argv[0]);
		return EXIT_FAILURE;
	}

	timeline_t tl;

	timeline_error_t err = timeline_load(&tl, argv[1]);
	if (err != TL_OK) {
		printf("Failed to Load Timeline %s\n", argv[1]);
		return EXIT_FAILURE;
	}

	puts("------------------------------------");
	puts("Parameters");
	puts("------------------------------------");
	world_print_params(&tl.params);

	world_t world;
	world.grid = (uint8_t *)calloc(tl.params.height * tl.params.width,
				       sizeof(*world.grid));
	world.params = tl.params;

	FILE *fp = fopen(argv[2], "w");
	if (!fp) {
		printf("Couldn't open %s\n", argv[2]);
	}
	fwrite(CSV_HEADER, strlen(CSV_HEADER), 1, fp);

	for (size_t i = 0; i < tl.saved_rounds; ++i) {
		err = timeline_get_round(&tl, world.grid);
		if (err == TL_END) {
			break;
		} else if (err != TL_OK) {
			puts("Failed to Read Round");
			break;
		}
		size_t healthy = world_get_healthy(&world);
		size_t infected = world_get_infected(&world);
		size_t immune = world_get_immune(&world);
		size_t dead = world_get_dead(&world);
		size_t total = healthy + infected + immune + dead;

		fprintf(fp, "%zu,%zu,%zu,%zu,%zu,%zu\n", i, healthy, infected,
			immune, dead, total);
	}
	fclose(fp);
	timeline_close(&tl);
	free(world.grid);
	return 0;
}
