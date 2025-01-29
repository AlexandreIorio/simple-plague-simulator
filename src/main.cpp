#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "timeline.h"
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include "world.h"

constexpr const char *parameterNames[] = {
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

#define MAX_TIMELINE_SIZE 500000000

bool generateParameterFile(const std::string &filename,
			   const world_parameters_t *w)
{
	std::ofstream file(filename);
	if (!file.is_open()) {
		std::cerr << "Error: Could not create file " << filename
			  << std::endl;
		return false;
	}

	file << parameterNames[0] << " " << w->population_percentage << "\n";
	file << parameterNames[1] << " " << w->healthy_infection_probability
	     << "\n";
	file << parameterNames[2] << " " << w->immune_infection_probability
	     << "\n";
	file << parameterNames[3] << " " << w->death_probability << "\n";
	file << parameterNames[4] << " " << w->initial_infected << "\n";
	file << parameterNames[5] << " " << w->initial_immune << "\n";
	file << parameterNames[6] << " " << w->proximity << "\n";
	file << parameterNames[7] << " " << w->height << "\n";
	file << parameterNames[8] << " " << w->width << "\n";
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
				w->population_percentage = value;
			else if (key == parameterNames[1])
				w->healthy_infection_probability = value;
			else if (key == parameterNames[2])
				w->immune_infection_probability = value;
			else if (key == parameterNames[3])
				w->death_probability = value;
			else if (key == parameterNames[4])
				w->initial_infected = value;
			else if (key == parameterNames[5])
				w->initial_immune = value;
			else if (key == parameterNames[6])
				w->proximity = value;
			else if (key == parameterNames[7])
				w->height = value;
			else if (key == parameterNames[8])
				w->width = value;
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
		<< w->population_percentage << "%)\n"
		<< "  -m, --population-immune <value>        Number of initial immune people (default: "
		<< w->initial_immune << "%)\n"
		<< "  -e, --healthy-infection <value>        Probability to infect a healthy person (default: "
		<< w->healthy_infection_probability << "%)\n"
		<< "  -n, --immune-infection <value>         Probability to infect an immune person (default: "
		<< w->immune_infection_probability << "%)\n"
		<< "  -d, --death-probability <value>         Probability of death (default: "
		<< w->death_probability << "%)\n"
		<< "  -i, --initial-infected <value>         Number of initial infected (default: "
		<< w->initial_infected << ")\n"
		<< "  -y, --proximity <value>                Proximity of infection (default: "
		<< w->proximity << ")\n"
		<< "  -h, --world-height <value>             Height of the grid (default: "
		<< w->height << ")\n"
		<< "  -w, --world-width <value>              Width of the grid (default: "
		<< w->width << ")\n"
		<< "  -r, --rounds <value>		     Max Rounds (default: No limit)\n"
		<< "  -f, --file <file>                      Load parameters from file\n"
		<< "  -g, --generate-file                    Generate a parameter file\n"
		<< "      --help                             Display this information\n";
}

int main(int argc, char *argv[])
{
	size_t total_rounds = 0;
	world_parameters_t params = {
		.height = 10,
		.width = 10,
		.initial_infected = 1,
		.initial_immune = 0,
		.proximity = 2,
		.population_percentage = 50,
		.death_probability = 10,
		.infection_duration = 5,
		.healthy_infection_probability = 10,
		.immune_infection_probability = 10,
	};

	const char *shortOptions = "p:e:n:d:r:i:m:y:h:w:f:g";
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
		{ "file", required_argument, nullptr, 'f' },
		{ "generate-file", no_argument, nullptr, 'g' },
		{ "help", no_argument, nullptr, ' ' },
		{ nullptr, 0, nullptr, 0 }
	};

	// args parsing
	int opt;
	while ((opt = getopt_long(argc, argv, shortOptions, longOptions,
				  nullptr)) != -1) {
		switch (opt) {
		case 'p':
			params.population_percentage = atoi(optarg);
			if (params.population_percentage > 100) {
				std::cerr
					<< "Error: Population must be > 0% and <=100%.\n";
				return 1;
			}
			break;

		case 'e':
			params.healthy_infection_probability = atoi(optarg);
			break;

		case 'n':
			params.immune_infection_probability = atoi(optarg);
			break;

		case 'd':
			params.death_probability = atoi(optarg);
			break;
		case 'r':
			total_rounds = atoi(optarg);
			break;
		case 'i':
			params.initial_infected = atoi(optarg);
			break;

		case 'm':
			params.initial_immune = atoi(optarg);
			break;

		case 'y':
			params.proximity = atoi(optarg);
			break;
		case 'h':
			params.height = atoi(optarg);
			break;

		case 'w':
			params.width = atoi(optarg);
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
		case '?':
		case ':':
		default:
			printUsage(&params);
			return 1;
		}
	}

	std::cout << "-----------------------------------\n";
	std::cout << "         Plague Simulator\n";
	std::cout << "-----------------------------------\n";
	std::cout << "Runtime : ";
#ifdef _CUDA
	std::cout << "CUDA";
#elif _OPENMP
	std::cout << "OpenMP";
#else
	std::cout << "CPU";
#endif
	std::cout << "\n";
	std::cout << "------------------------------------\n";
	std::cout << "Parameters\n";
	std::cout << "------------------------------------\n";
	std::cout
		<< "Population                  : "
		<< params.population_percentage << " %\n"
		<< "World height                : " << params.height << "\n"
		<< "World Width                 : " << params.width << "\n"
		<< "World size                  : "
		<< params.height * params.width << "\n"
		<< "Proximity                   : " << params.proximity << "\n"
		<< "Infection duration          : " << params.infection_duration
		<< " turns\n"
		<< "Healthy infection probability:"
		<< params.healthy_infection_probability << " % \n"
		<< "Immune infection probability: "
		<< params.immune_infection_probability << " % \n"
		<< "Death probability           : " << params.death_probability
		<< " %\n"
		<< "Initial infected            : " << params.initial_infected
		<< "\n"
		<< "Population immunized        : " << params.initial_immune
		<< " %\n";
	std::cout << "\n";
	std::cout << "-----------------------------------\n";
	std::cout << "         Initialisation\n";
	std::cout << "-----------------------------------\n";

	std::cout << "Initializing World ..." << std::endl;
	struct timespec start, finish;

	world_t world;
	clock_gettime(CLOCK_MONOTONIC, &start);
	int ret = world_init(&world, &params);

	if (ret < 0) {
		return ret;
	}
	clock_gettime(CLOCK_MONOTONIC, &finish);

	double init_elapsed = (finish.tv_sec - start.tv_sec);
	init_elapsed += (finish.tv_nsec - start.tv_nsec) / 1e9;

	std::cout << "Initialization Duration: " << init_elapsed << " s"
		  << std::endl;

	int initialImmune = world_get_immune(&world);

	std::cout << "------------------------------------\n";
	std::cout << "Initial world :\n";
	std::cout << "------------------------------------\n";
	std::cout << "Number of healthy people  : " << world_get_healthy(&world)
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

	timeline_error_t tl_err;
	double total_elapsed = 0;
	timeline_t tl;

	tl_err = timeline_init(&tl, &world.params, "timeline.bin",
			       MAX_TIMELINE_SIZE);
	if (tl_err != TL_OK) {
		std::cerr << "Failed to allocate memory to store world state"
			  << '\n';
		return 1;
	}
	bool tl_max_size_reached = false;

	size_t nb_rounds = 0;
	while (world_get_infected(&world) > 0) {
		if (total_rounds > 0 && nb_rounds >= total_rounds) {
			break;
		}

		if (nb_rounds % 10 == 0) {
			std::cout << "Round " << nb_rounds << '\n';
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
		++nb_rounds;

		if (!tl_max_size_reached) {
			tl_err =
				timeline_push_round(&tl, (uint8_t *)world.grid);

			if (tl_err == TL_MAX_SIZE) {
				tl_max_size_reached = true;
				std::cerr
					<< "Max Timeline size reached at round "
					<< nb_rounds
					<< ". Extra rounds won't be saved\n";

			} else if (tl_err != TL_OK) {
				std::cerr << "Failed to save last round\n";
				break;
			}
		}
	}
	std::cout << "------------------------------------\n";
	std::cout << "Saving Timeline\n";

	tl_err = timeline_save(&tl);
	if (tl_err != TL_OK) {
		std::cout << "Failed to create timeline\n";
	} else {
		std::cout << "Timeline created\n";
	}

	const double total_time = init_elapsed + total_elapsed;
	std::cout << "Initialization took       : " << init_elapsed << " s\n";
	std::cout << "Simulation took           : " << total_elapsed << " s\n";
	std::cout << "Total Time                : " << total_time << " s\n";
	std::cout << "Number of turns           : " << nb_rounds << '\n';
	std::cout << "Number of healthy people  : " << world_get_healthy(&world)
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
