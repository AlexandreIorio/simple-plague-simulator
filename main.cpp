#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "world.h"
#include "video_utils.h"

static const char *parameterNames[] = {
	"population",
	"healthy_infection_probability",
	"immune_infection_probability",
	"death_probability",
	"initial_infected",
	"initial_immune",
	"proximity",
	"world_height",
	"world_width",
};

bool generateParameterFile(const std::string &filename,
			   const world_parameters_t *w)
{
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error: Could not create file " << filename
			  << std::endl;
		return false;
	}

	file << parameterNames[0] << " " << w->populationPercent << "\n";
	file << parameterNames[1] << " " << w->healthyInfectionProbability
	     << "\n";
	file << parameterNames[2] << " " << w->immuneInfectionProbability
	     << "\n";
	file << parameterNames[3] << " " << w->deathProbability << "\n";
	file << parameterNames[4] << " " << w->initialInfected << "\n";
	file << parameterNames[5] << " " << w->initialImmune << "\n";
	file << parameterNames[6] << " " << w->proximity << "\n";
	file << parameterNames[7] << " " << w->worldHeight << "\n";
	file << parameterNames[8] << " " << w->worldWidth << "\n";

	file.close();
	std::cout << "Parameters written to " << filename << std::endl;
	return true;
}

bool loadParametersFromFile(const std::string &filename, world_parameters_t *w)
{
	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error: Could not open file " << filename
			  << std::endl;
		return false;
	}

	std::string line;
	while (std::getline(file, line)) {
		std::istringstream iss(line);
		std::string key;
		int value;
		if (iss >> key >> value) {
			if (key == parameterNames[0])
				w->populationPercent = value;
			else if (key == parameterNames[1])
				w->healthyInfectionProbability = value;
			else if (key == parameterNames[2])
				w->immuneInfectionProbability = value;
			else if (key == parameterNames[3])
				w->deathProbability = value;
			else if (key == parameterNames[4])
				w->initialInfected = value;
			else if (key == parameterNames[5])
				w->initialImmune = value;
			else if (key == parameterNames[6])
				w->proximity = value;
			else if (key == parameterNames[7])
				w->worldHeight = value;
			else if (key == parameterNames[8])
				w->worldWidth = value;
			else {
				std::cerr << "Warning: Unknown key '" << key
					  << "' in file " << filename
					  << std::endl;
			}
		}
	}
	file.close();
	return true;
}

void printUsage(const world_parameters_t *w)
{
	std::cout
		<< "Usage: Plague Simulator [options]\n"
		<< "Options:\n"
		<< "  -p, --population <value>               Percent of population (default: "
		<< w->populationPercent << "%)\n"
		<< "  -m, --population-immune <value>        Number of initial immune people (default: "
		<< w->initialImmune << "%)\n"
		<< "  -e, --healthy-infection <value>        Probability to infect a healthy person (default: "
		<< w->healthyInfectionProbability << "%)\n"
		<< "  -n, --immune-infection <value>         Probability to infect an immune person (default: "
		<< w->immuneInfectionProbability << "%)\n"
		<< "  -d, --death-probability <value>         Probability of death (default: "
		<< w->deathProbability << "%)\n"
		<< "  -i, --initial-infected <value>         Number of initial infected (default: "
		<< w->initialInfected << ")\n"
		<< "  -y, --proximity <value>                Proximity of infection (default: "
		<< w->proximity << ")\n"
		<< "  -h, --world-height <value>             Height of the grid (default: "
		<< w->worldHeight << ")\n"
		<< "  -w, --world-width <value>              Width of the grid (default: "
		<< w->worldWidth << ")\n"
		<< "  -r, --rounds <value>		     Max Rounds (default: No limit)\n"
		<< "  -f, --file <file>                      Load parameters from file\n"
		<< "  -g, --generate-file                    Generate a parameter file\n"
		<< "      --help                             Display this information\n";
}

int main(int argc, char *argv[])
{
	bool doVideo = false;
	size_t total_rounds = 0;
	world_parameters_t params = {
		.worldHeight = 10,
		.worldWidth = 10,
		.populationPercent = 50,
		.initialInfected = 1,
		.initialImmune = 0,
		.deathProbability = 10,
		.infectionDuration = 10,
		.healthyInfectionProbability = 10,
		.immuneInfectionProbability = 10,
		.proximity = 2,
	};

	const char *shortOptions = "p:e:n:d:r:i:m:y:h:w:vf:g";
	const struct option longOptions[] = {
		{ "population", required_argument, nullptr, 'p' },
		{ "healthy-infection-probability", required_argument, nullptr,
		  'e' },
		{ "immune-infection-probability", required_argument, nullptr,
		  'n' },
		{ "death-probability", required_argument, nullptr, 'd' },
		{ "rounds", required_argument, nullptr, 'r' },
		{ "initial-infected", required_argument, nullptr, 'i' },
		{ "initial-immune", required_argument, nullptr, 'm' },
		{ "proximity", required_argument, nullptr, 'y' },
		{ "world-height", required_argument, nullptr, 'h' },
		{ "world-width", required_argument, nullptr, 'w' },
		{ "video", no_argument, nullptr, 'v' },
		{ "file", required_argument, nullptr, 'f' },
		{ "generate-file", no_argument, nullptr,
		  'g' }, // Nouvelle option
		{ "help", no_argument, nullptr, ' ' },
		{ nullptr, 0, nullptr, 0 }
	};

	// args parsing
	int opt;
	while ((opt = getopt_long(argc, argv, shortOptions, longOptions,
				  nullptr)) != -1) {
		switch (opt) {
		case 'p':
			params.populationPercent = atoi(optarg);
			if (params.populationPercent > 100) {
				std::cerr
					<< "Error: Population must be > 0% and <=100%.\n";
				return 1;
			}
			break;

		case 'e':
			params.healthyInfectionProbability = atoi(optarg);
			break;

		case 'n':
			params.immuneInfectionProbability = atoi(optarg);
			break;

		case 'd':
			params.deathProbability = atoi(optarg);
			break;

		case 'r':
			params.deathProbability = atoi(optarg);
			break;
		case 'i':
			params.initialInfected = atoi(optarg);
			break;

		case 'm':
			params.initialImmune = atoi(optarg);
			break;

		case 'y':
			params.proximity = atoi(optarg);
			break;
		case 'h':
			params.worldHeight = atoi(optarg);
			break;

		case 'w':
			params.worldWidth = atoi(optarg);
			break;
		case 'f': {
			std::string filename = optarg;
			if (!loadParametersFromFile(filename, &params)) {
				return 1;
			}
			break;
		}
		case 'g': {
			std::string filename = "parameters.txt";
			if (!generateParameterFile(filename, &params)) {
				return 1;
			}
			exit(0);
		}
		case 'v':
			doVideo = true;
			break;

		case '?':
		case ':':
		default:
			printUsage(&params);
			return 1;
		}
	}

	world_t world;
	int ret = world_init(&world, &params);
	if (ret < 0) {
		return ret;
	}

	std::cout << "-----------------------------------\n";
	std::cout << "         Plague Simulator\n";
	std::cout << "-----------------------------------\n";
	std::cout << "------------------------------------\n";
	std::cout << "Parameters :\n";
	std::cout << "------------------------------------\n";
	std::cout
		<< "Population                  : " << params.populationPercent
		<< "%\n"
		<< "World height                : " << params.worldHeight
		<< "\n"
		<< "World Width                 : " << params.worldWidth << "\n"
		<< "World size                  : "
		<< params.worldHeight * params.worldWidth << "\n"
		<< "Proximity                   : " << params.proximity << "\n"
		<< "Infection duration          : " << params.infectionDuration
		<< " turns\n"
		<< "Healty infection probability: "
		<< params.healthyInfectionProbability << " % \n"
		<< "Immune infection probability: "
		<< params.immuneInfectionProbability << " % \n"
		<< "Death probability           : " << params.deathProbability
		<< "%\n"
		<< "Initial infected            : " << params.initialInfected
		<< "\n"
		<< "Population immunized        : " << params.initialImmune
		<< "%\n";

	std::cout << "\n";
	int initialImmune = world_get_immune(&world);

	std::cout << "------------------------------------\n";
	std::cout << "Initial world :\n";
	std::cout << "------------------------------------\n";
	std::cout << "Number of healty people   : " << world_get_healthy(&world)
		  << '\n';
	std::cout
		<< "Number of infected people : " << world_get_infected(&world)
		<< '\n';
	std::cout << "Number of immunized people: " << initialImmune << '\n';
	std::cout << "\n";
	std::cout << "------------------------------------\n";
	std::cout << "Simulation\n";
	std::cout << "------------------------------------\n";
	std::cout << "Simulation started\n";
	struct timespec start, finish;

	double total_elapsed = 0;
	size_t rounds = 0;
	if (doVideo) {
		int **grids = NULL;
		size_t grids_size = 0;
		if (total_rounds > 0) {
			grids_size = total_rounds;
		} else {
			grids_size = 128;
		}

		grids = (int **)calloc(grids_size, sizeof(*grids));

		if (!grids) {
			std::cerr
				<< "Failed to allocate memory to store world state"
				<< '\n';
			return 1;
		}

		while (world_get_infected(&world) > 0) {
			if (total_rounds > 0 && rounds >= total_rounds) {
				break;
			}

			if (rounds >= grids_size) {
				std::cout << "Round " << rounds << '\n';
				const size_t new_size = grids_size * 2;
				int **new_grids = (int **)std::realloc(
					grids, grids_size * params.worldWidth *
						       params.worldHeight *
						       sizeof(*world.grid));
				if (!new_grids) {
					std::cerr
						<< "Failed to allocate more memory to store world state"
						<< '\n';
					break;
				}
				grids_size *= new_size;
				grids = new_grids;
			}

			void *ret = world_prepare_update(&world);
			if (!ret) {
				std::cerr << "Failed to Update World" << '\n';
				break;
			}

			clock_gettime(CLOCK_MONOTONIC, &start);
			world_update(&world, ret);
			clock_gettime(CLOCK_MONOTONIC, &finish);

			double round_elapsed = (finish.tv_sec - start.tv_sec);
			round_elapsed += (finish.tv_nsec - start.tv_nsec) / 1e9;

			total_elapsed += round_elapsed;

			grids[rounds] = (int *)malloc(params.worldWidth *
						      params.worldHeight *
						      sizeof(**grids));
			if (!grids[rounds]) {
				std::cerr
					<< "Failed to allocate more memory to store world state"
					<< '\n';
				break;
			}
			std::memcpy(grids[rounds], world.grid,
				    params.worldWidth * params.worldHeight *
					    sizeof(*grids[rounds]));
			++rounds;
		}
		std::cout << "------------------------------------\n";
		std::cout << "Creating video\n";
		std::cout << "------------------------------------\n";
		std::cout << "Creating...\n";
		create_video(grids, rounds, params.worldWidth,
			     params.worldHeight, "plague.avi", 20, 10);
		free(grids);
		std::cout << "Video created\n";
	} else {
		while (world_get_infected(&world) > 0) {
			if (total_rounds > 0 && rounds >= total_rounds) {
				break;
			}
			void *ret = world_prepare_update(&world);
			if (!ret) {
				std::cerr << "Failed to Update World" << '\n';
			}

			clock_gettime(CLOCK_MONOTONIC, &start);
			world_update(&world, ret);
			clock_gettime(CLOCK_MONOTONIC, &finish);

			double round_elapsed = (finish.tv_sec - start.tv_sec);
			round_elapsed += (finish.tv_nsec - start.tv_nsec) / 1e9;
			total_elapsed += round_elapsed;
		}
	}

	std::cout << "Simulation took           : " << total_elapsed << " s\n";
	std::cout << "Number of turns           : " << rounds << '\n';
	std::cout << "Number of healty people   : " << world_get_healthy(&world)
		  << '\n';
	std::cout << "Number of immunized people: " << world_get_immune(&world)
		  << '\n';
	std::cout << "Number of survivor        : "
		  << world_get_immune(&world) + world_get_healthy(&world)
		  << '\n';
	std::cout << "Number of dead people     : " << world_get_dead(&world)
		  << '\n';
	std::cout << "\n";
	world_destroy(&world);
	return 0;
}
