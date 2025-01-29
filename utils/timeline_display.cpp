#include <SDL2/SDL.h>
#include <SDL2/SDL_video.h>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include "timeline.h"

const int SCREEN_WIDTH = 1024;
const int SCREEN_HEIGHT = 1024;
const int TARGET_FPS = 60;
const int FRAME_DELAY = 1000 / TARGET_FPS;
SDL_Window *window = nullptr;
SDL_Renderer *renderer = nullptr;

static bool sdl_init()
{
	if (SDL_Init(SDL_INIT_VIDEO) < 0) {
		std::cerr << "SDL could not initialize! SDL_Error: "
			  << SDL_GetError() << std::endl;
		return false;
	}

	window = SDL_CreateWindow("Plague Simulator", SDL_WINDOWPOS_UNDEFINED,
				  SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH,
				  SCREEN_HEIGHT, SDL_WINDOW_BORDERLESS);
	if (!window) {
		std::cerr << "Window could not be created! SDL_Error: "
			  << SDL_GetError() << std::endl;
		return false;
	}

	renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
	if (!renderer) {
		std::cerr << "Renderer could not be created! SDL_Error: "
			  << SDL_GetError() << std::endl;
		return false;
	}

	return true;
}

static void sdl_close()
{
	SDL_DestroyRenderer(renderer);
	SDL_DestroyWindow(window);
	SDL_Quit();
}
static void draw(uint8_t *grid, size_t width, size_t height)
{
	SDL_SetRenderDrawColor(renderer, 0, 0, 0,
			       255); // Clear screen with black
	SDL_RenderClear(renderer);

	int cell_width = SCREEN_WIDTH / width;
	int cell_height = SCREEN_HEIGHT / height;
	if (cell_width == 0) {
		cell_width = 1;
	}
	if (cell_height == 0) {
		cell_height = 1;
	}
	for (size_t y = 0; y < height; ++y) {
		for (size_t x = 0; x < width; ++x) {
			uint8_t cell = grid[y * width + x];
			SDL_Rect rect{ .x = static_cast<int>(x * cell_width),
				       .y = static_cast<int>(y * cell_height),
				       .w = cell_width,
				       .h = cell_height };

			switch (cell) {
			case 0:
				SDL_SetRenderDrawColor(renderer, 200, 200, 200,
						       255); // Grayish (empty)
				break;
			case 1:
				SDL_SetRenderDrawColor(renderer, 0, 255, 0,
						       255); // Green (healthy)
				break;
			case 2:
				SDL_SetRenderDrawColor(renderer, 255, 0, 0,
						       255); // Red (infected)
				break;
			case 3:
				SDL_SetRenderDrawColor(renderer, 0, 0, 0,
						       255); // Black (dead)
				break;
			case 4:
				SDL_SetRenderDrawColor(renderer, 0, 255, 255,
						       255); // Cyan (immune)
				break;
			default:
				SDL_SetRenderDrawColor(renderer, 255, 255, 255,
						       255); // White (unknown)
				break;
			}

			SDL_RenderFillRect(renderer, &rect);
		}
	}

	SDL_RenderPresent(renderer);
}

int main(int argc, char *argv[])
{
	if (argc < 2) {
		std::cerr << "Usage " << argv[0] << " <timeline_file.bin>\n";
		return EXIT_FAILURE;
	}

	bool quit = false;
	SDL_Event e;
	timeline_t tl;

	timeline_error_t err = timeline_load(&tl, argv[1]);
	if (err != TL_OK) {
		std::cerr << "Failed to Load Timeline " << argv[1] << '\n';
		return EXIT_FAILURE;
	}

	if (!sdl_init()) {
		return EXIT_FAILURE;
	}
	std::cout << "------------------------------------\n";
	std::cout << "Parameters\n";
	std::cout << "------------------------------------\n";
	world_print_params(&tl.params);
	std::cout << "\n";

	uint8_t *grid = new uint8_t[tl.params.height * tl.params.width];

	while (!quit) {
		const uint32_t frame_start = SDL_GetTicks();
		while (SDL_PollEvent(&e)) {
			if (e.type == SDL_QUIT) {
				quit = true;
			}
		}
		err = timeline_get_round(&tl, grid);
		if (err == TL_END) {
			break;
		} else if (err != TL_OK) {
			std::cerr << "Failed to Read Round\n";
			break;
		}

		draw(grid, tl.params.width, tl.params.height);
		const uint32_t frame_time = SDL_GetTicks() - frame_start;
		if (frame_time < FRAME_DELAY) {
			SDL_Delay(FRAME_DELAY - frame_time);
		}
	}

	timeline_close(&tl);
	sdl_close();
	delete[] grid;
	return 0;
}
