#include <getopt.h>
#include "plague.h"
#include "video_utils.h"

using namespace std;
void startTimer(timespec &start)
{
	clock_gettime(CLOCK_MONOTONIC, &start);
}

double stopTimer(const timespec &start)
{
	timespec end;
	clock_gettime(CLOCK_MONOTONIC, &end);

	// Calcul de la différence en ms
	double elapsed = (end.tv_sec - start.tv_sec) +
			 (end.tv_nsec - start.tv_nsec) / 1e6;
	return elapsed;
}

void printUsage(Plague p)
{
	cout << "Usages: Plague Simulator [options]\n"
	     << "Options:\n"
	     << "  -p, --population     Percent of population, default "
	     << p.populationPercent << "%\n"
	     << "  -m, --popilation-immune       Number of initial immune, default "
	     << p.immunePercent << "%\n"
	     << "  -e, --heathy-infection       Probability to infect a healthy person , default "
	     << p.healthyInfectionProbability << "\n"
	     << "  -n, --immune-infection       Probability to infect an immune person , default "
	     << p.immuneInfectionProbability << "\n"
	     << "  -r, --dead-probability       Probability of death, default "
	     << p.deathProbability << "\n"
	     << "  -i, --initial-infected       Number of initial infected, default "
	     << p.initialInfected << "\n"
	     << "  -y, --proximity      Proximity of infection, default "
	     << p.proximity << "\n"
	     << "  -h, --world-height  height of the grid, default "
	     << p.worldHeight << "\n"
	     << "  -w, --world-width   width of the grid, default "
	     << p.worldWidth << "\n"
	     << "--help           Display this information\n";
}

int main(int argc, char *argv[])
{
	Plague p;
	bool doVideo = false;

	p.worldHeight = 10;
	p.worldWidth = 10;
	p.populationPercent = 50;
	p.immunePercent = 0;
	p.deathProbability = 10;
	p.infectionDuration = 10;
	p.healthyInfectionProbability = 10;
	p.immuneInfectionProbability = 10;
	p.initialInfected = 1;
	p.proximity = 2;

	const char *shortOptions = "p:e:n:d:r:i:m:y:h:w:v:";
	const struct option longOptions[] = {
		{ "population", required_argument, nullptr, 'p' },
		{ "healthy-infection-probability", required_argument, nullptr,
		  'e' },
		{ "immune-infection-probability", required_argument, nullptr,
		  'n' },
		{ "dead-probability", required_argument, nullptr, 'r' },
		{ "initial-infected", required_argument, nullptr, 'i' },
		{ "initial-immune", required_argument, nullptr, 'm' },
		{ "proximity", required_argument, nullptr, 'y' },
		{ "world-height", required_argument, nullptr, 'h' },
		{ "world-width", required_argument, nullptr, 'w' },
		{ "video", no_argument, nullptr, 'v' },
		{ "help", no_argument, nullptr, ' ' },
		{ nullptr, 0, nullptr, 0 }
	};

	// args parsing
	int opt;
	while ((opt = getopt_long(argc, argv, shortOptions, longOptions,
				  nullptr)) != -1) {
		switch (opt) {
		case 'p':
			p.populationPercent = atoi(optarg);
			if (p.populationPercent < 0 ||
			    p.populationPercent > 100) {
				cerr << "Error: Population must be > 0% and <=100%.\n";
				return 1;
			}
			break;

		case 'e':
			p.healthyInfectionProbability = atoi(optarg);
			break;

		case 'n':
			p.immuneInfectionProbability = atoi(optarg);
			break;

		case 'd':
			p.healthyInfectionProbability = atoi(optarg);
			break;

		case 'r':
			p.deathProbability = atoi(optarg);
			break;
		case 'i':
			p.initialInfected = atoi(optarg);
			break;

		case 'm':
			p.immunePercent = atoi(optarg);
			break;

		case 'y':
			p.proximity = atoi(optarg);
			break;
		case 'h':
			p.worldHeight = atoi(optarg);
			break;

		case 'w':
			p.worldWidth = atoi(optarg);
			break;
		case 'v':
			doVideo = true;
			break;

		case '?':
		case ':':
		default:
			printUsage(p);
			return 1;
		}
	}

	p.world = std::vector<vector<int> >(p.worldHeight,
					    vector<int>(p.worldWidth, EMPTY));
	p.infectionDurationMap = std::vector<vector<int> >(
		p.worldHeight, vector<int>(p.worldWidth, p.infectionDuration));

	initializeGrid(p);
	initializeImmune(p);
	initializeInfection(p);

	cout << "-----------------------------------\n";
	cout << "         Plague Simulator\n";
	cout << "-----------------------------------\n";
	cout << "------------------------------------\n";
	cout << "Parameters :\n";
	cout << "------------------------------------\n";
	cout << "Population                  : " << p.populationPercent << "%\n"
	     << "Population immunized        : " << p.immunePercent << "%\n"
	     << "World height                : " << p.worldHeight << "\n"
	     << "World Width                 : " << p.worldWidth << "\n"
	     << "World size                  : " << p.worldHeight * p.worldWidth
	     << "\n"
	     << "Proximity                   : " << p.proximity << "\n"
	     << "Infection duration          : " << p.infectionDuration
	     << " turns\n"
	     << "Healty infection probability: "
	     << p.healthyInfectionProbability << " % \n"
	     << "Immune infection probability: " << p.immuneInfectionProbability
	     << " % \n"
	     << "Death probability           : " << p.deathProbability << "%\n"
	     << "Initial infected            : " << p.initialInfected << "\n";

	cout << "\n";
	int initialImmune = getNbImmune(p);
	cout << "------------------------------------\n";
	cout << "Initial world :\n";
	cout << "------------------------------------\n";
	cout << "Number of healty people   : " << getNbHealty(p) << endl;
	cout << "Number of infected people : " << getNbInfected(p) << endl;
	cout << "Number of immunized people: " << initialImmune << endl;
	cout << "\n";
	cout << "------------------------------------\n";
	cout << "Simulation\n";
	cout << "------------------------------------\n";
	cout << "Simulation started\n";
	struct timespec start, finish;

	clock_gettime(CLOCK_MONOTONIC, &start);

	vector<vector<vector<int> > > stepsToApocalypse;
	int nb_turn = 0;

	vector<vector<int> > stepToApocalypse = p.world;
	stepsToApocalypse.push_back(stepToApocalypse);
	//init timer
	while (getNbInfected(p) > 0) {
		updateWorld(p);
		nb_turn++;
		stepToApocalypse = p.world;
		stepsToApocalypse.push_back(stepToApocalypse);
	}

	double elapsed;
	clock_gettime(CLOCK_MONOTONIC, &finish);
	elapsed = (finish.tv_sec - start.tv_sec);
	elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
	cout << "Simulation took           : " << elapsed << " s\n";
	cout << "Number of turns           : " << nb_turn << endl;
	cout << "Number of healty people   : " << getNbHealty(p) << endl;
	cout << "Number of immunized people: " << getNbImmune(p) << endl;
	cout << "Number of survivor        : " << getNbImmune(p) - initialImmune
	     << endl;
	cout << "Number of dead people     : " << getNbDead(p) << endl;
	cout << "\n";

	if (doVideo) {
		cout << "------------------------------------\n";
		cout << "Creating video\n";
		cout << "------------------------------------\n";
		cout << "Creating...\n";
		createVideo(stepsToApocalypse, "plague.avi", 20, 10);
		cout << "Video created\n";
	}
	return 0;
}
