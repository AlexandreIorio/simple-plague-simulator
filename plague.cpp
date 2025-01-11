
#include <iostream>
#include <vector>
#include <getopt.h>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include <opencv2/opencv.hpp>

using namespace std;

enum State { EMPTY = 0, HEALTHY, INFECTED, DEAD, IMMUNE };

int worldHeight = 10;
int worldWidth = 10;
int populationPercent = 50;
int exposureDuration = 5;
int deathProbability = 10;
int infectionDuration = 10;
int initialInfected = 1;
int initialImmune = 0;
int proximity = 2;
std::vector<vector<int> > world;
std::vector<vector<int> > exposureDurationMap;
std::vector<vector<int> > infectionDurationMap;

cv::Scalar stateToColor(State state)
{
	switch (state) {
	case EMPTY:
		return cv::Scalar(255, 255, 255); // Blanc
	case HEALTHY:
		return cv::Scalar(0, 255, 0); // Vert
	case INFECTED:
		return cv::Scalar(0, 0, 255); // Rouge
	case DEAD:
		return cv::Scalar(0, 0, 0); // Noir
	case IMMUNE:
		return cv::Scalar(255, 255, 0); // Jaune
	default:
		return cv::Scalar(255, 255, 255); // Par défaut blanc
	}
}

// Fonction pour créer une image à partir d'une grille 2D
cv::Mat createFrame(const std::vector<std::vector<int> > &grid, int cellSize)
{
	int rows = grid.size();
	int cols = grid[0].size();

	// Créer une image de taille adaptée
	cv::Mat frame(rows * cellSize, cols * cellSize, CV_8UC3,
		      cv::Scalar(255, 255, 255));

	// Remplir les cellules avec les couleurs correspondantes
	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			cv::Rect cellRect(j * cellSize, i * cellSize, cellSize,
					  cellSize);
			cv::Scalar color =
				stateToColor(static_cast<State>(grid[i][j]));
			cv::rectangle(frame, cellRect, color, cv::FILLED);
		}
	}

	return frame;
}

// Fonction pour créer une vidéo à partir d'une grille 3D
void createVideo(const std::vector<std::vector<std::vector<int> > > &grid3D,
		 const std::string &outputPath, int cellSize = 20, int fps = 10)
{
	if (grid3D.empty() || grid3D[0].empty() || grid3D[0][0].empty()) {
		throw std::invalid_argument("Grid 3D is empty or invalid");
	}

	int rows = grid3D[0].size();
	int cols = grid3D[0][0].size();

	// Définir les dimensions de la vidéo
	cv::Size videoSize(cols * cellSize, rows * cellSize);

	// Initialiser le writer vidéo
	cv::VideoWriter writer(outputPath,
			       cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps,
			       videoSize);
	if (!writer.isOpened()) {
		throw std::runtime_error(
			"Could not open the video file for writing");
	}

	// Générer chaque frame
	for (const auto &grid : grid3D) {
		cv::Mat frame = createFrame(grid, cellSize);
		writer.write(frame);
	}

	writer.release();
}

///@biref Init the grid with random values, related to the populationPercent
///@param grid The grid to initialize
///@param populationPercent The percentage of the population that will cover the world
///@note non-parralelizable
void initializeGrid()
{
	srand(time(0));
	int maxPeople = worldHeight * worldWidth * populationPercent / 100;
	int people = 0;

	while (people < maxPeople) {
		for (int i = 0; i < worldHeight; i++) {
			for (int j = 0; j < worldWidth; j++) {
				if (world[i][j] != EMPTY) {
					continue;
				};
				if (people >= maxPeople) {
					break;
				}

				if (rand() % 100 < populationPercent) {
					world[i][j] = HEALTHY;
					people++;
				} else {
					world[i][j] = EMPTY;
				}
			}
		}
	}
}

void getHealty(vector<vector<int> > &healthyPeopleIndexes)
{
	for (int i = 0; i < worldHeight; i++) {
		for (int j = 0; j < worldWidth; j++) {
			if (world[i][j] == HEALTHY) {
				healthyPeopleIndexes.push_back({ i, j });
			}
		}
	}
}

void getInfected(vector<vector<int> > &infectedPeopleIndexes)
{
	for (int i = 0; i < worldHeight; i++) {
		for (int j = 0; j < worldWidth; j++) {
			if (world[i][j] == INFECTED) {
				infectedPeopleIndexes.push_back({ i, j });
			}
		}
	}
}

int getNbInfected()
{
	int nbInfected = 0;
	for (int i = 0; i < worldHeight; i++) {
		for (int j = 0; j < worldWidth; j++) {
			if (world[i][j] == INFECTED) {
				nbInfected++;
			}
		}
	}
	return nbInfected;
}

int getNbHealty()
{
	int nbHealty = 0;
	for (int i = 0; i < worldHeight; i++) {
		for (int j = 0; j < worldWidth; j++) {
			if (world[i][j] == HEALTHY) {
				nbHealty++;
			}
		}
	}
	return nbHealty;
}

int getNbDead()
{
	int nbDead = 0;
	for (int i = 0; i < worldHeight; i++) {
		for (int j = 0; j < worldWidth; j++) {
			if (world[i][j] == DEAD) {
				nbDead++;
			}
		}
	}
	return nbDead;
}

int getNbEmpty()
{
	int nbEmpty = 0;
	for (int i = 0; i < worldHeight; i++) {
		for (int j = 0; j < worldWidth; j++) {
			if (world[i][j] == EMPTY) {
				nbEmpty++;
			}
		}
	}
	return nbEmpty;
}

///@brief Infects randomly a number of healty people

void initializeInfection(int nbInfected)
{
	// Récupérer toutes les cellules comme une liste plate
	std::vector<std::pair<int, int> > cells;
	for (int i = 0; i < worldHeight; ++i) {
		for (int j = 0; j < worldWidth; ++j) {
			cells.emplace_back(
				i, j); // Ajouter chaque cellule à la liste
		}
	}

	// Mélanger les cellules aléatoirement
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(cells.begin(), cells.end(), g);

	// Infecter le nombre souhaité de cellules
	int infected = 0;
	for (const auto &cell : cells) {
		int i = cell.first;
		int j = cell.second;

		if (world[i][j] == HEALTHY) {
			world[i][j] = INFECTED;
			++infected;

			if (infected >= nbInfected) {
				break; // Arrêter lorsque le nombre requis est atteint
			}
		}
	}
}

void initializeImmune(int nbImmune)
{
	// Récupérer toutes les cellules comme une liste plate
	std::vector<std::pair<int, int> > cells;
	for (int i = 0; i < worldHeight; ++i) {
		for (int j = 0; j < worldWidth; ++j) {
			cells.emplace_back(
				i, j); // Ajouter chaque cellule à la liste
		}
	}

	// Mélanger les cellules aléatoirement
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(cells.begin(), cells.end(), g);

	int immune = 0;
	for (const auto &cell : cells) {
		int i = cell.first;
		int j = cell.second;

		if (world[i][j] == HEALTHY) {
			world[i][j] = IMMUNE;
			++immune;

			if (immune >= nbImmune) {
				break; // Arrêter lorsque le nombre requis est atteint
			}
		}
	}
}

// display world
void displayWorld()
{
	for (int i = 0; i < worldHeight; i++) {
		for (int j = 0; j < worldWidth; j++) {
			switch (world[i][j]) {
			case EMPTY:
				cout << ".";
				break;
			case HEALTHY:
				cout << "H";
				break;
			case INFECTED:
				cout << "X";
				break;
			case DEAD:
				cout << "D";
				break;
			}
		}
		cout << endl;
	}
	cout << endl;
}

void getNeighbours(int i, int j, vector<vector<int> > &neighbours)
{
	for (int dx = -proximity; dx <= proximity; ++dx) {
		for (int dy = -proximity; dy <= proximity; ++dy) {
			if (dx == 0 && dy == 0) {
				continue;
			}

			int ni = i + dx;
			int nj = j + dy;

			// Vérification des limites de la grille
			if (ni >= 0 && ni < worldHeight && nj >= 0 &&
			    nj < worldWidth) {
				if (world[ni][nj] != EMPTY) {
					neighbours.push_back({ ni, nj });
				}
			}
		}
	}
}

void updateWorld()
{
	vector<vector<int> > tmpWorld(worldHeight,
				      vector<int>(worldWidth, EMPTY));
	for (int i = 0; i < worldHeight; i++) {
		for (int j = 0; j < worldWidth; j++) {
			tmpWorld[i][j] = world[i][j];
		}
	}

	for (int i = 0; i < worldHeight; i++) {
		for (int j = 0; j < worldWidth; j++) {
			if (world[i][j] == INFECTED) {
				if (infectionDurationMap[i][j] == 0) {
					if (rand() % 100 < deathProbability) {
						tmpWorld[i][j] = DEAD;
					} else {
						tmpWorld[i][j] = IMMUNE;
					}
				} else {
					infectionDurationMap[i][j]--;
				}
			} else if (world[i][j] == HEALTHY) {
				vector<vector<int> > neighbours;
				getNeighbours(i, j, neighbours);
				for (auto &neighbour : neighbours) {
					if (world[neighbour[0]][neighbour[1]] ==
					    INFECTED) {
						if (exposureDurationMap[i][j] >
						    0) {
							exposureDurationMap[i]
									   [j]--;
						} else {
							tmpWorld[i][j] =
								INFECTED;
						}
						break;
					}
				}
			}
		}
	}

	for (int i = 0; i < worldHeight; i++) {
		for (int j = 0; j < worldWidth; j++) {
			world[i][j] = tmpWorld[i][j];
		}
	}
}

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
			 (end.tv_nsec - start.tv_nsec) / 1e3;
	return elapsed;
}

void printUsage()
{
	cout << "Usages: Plague Simulator [options]\n"
	     << "Options:\n"
	     << "  -p, --population     Percent of population, default"
	     << populationPercent << "%\n"
	     << "  -e, --exposure       Duration of exposure before infection, default"
	     << exposureDuration << "\n"
	     << "  -d, --duration       Duration of infection before immune or death, default"
	     << infectionDuration << "\n"
	     << "  -r, --dead-probability       Probability of death, default"
	     << deathProbability << "\n"
	     << "  -i, --initial-infected       Number of initial infected, default"
	     << initialInfected << "\n"
	     << "  -m, --initial-immune       Number of initial immune, default"
	     << initialImmune << "\n"
	     << "  -y, --proximity      Proximity of infection, default"
	     << proximity << "\n"
	     << "  -h, --world-height  height of the grid, default"
	     << worldHeight << "\n"
	     << "  -w, --world-width   width of the grid, default" << worldWidth
	     << "\n"
	     << "--help           Display this information\n";
}

int main(int argc, char *argv[])
{
	// Définir les options longues et courtes
	const char *shortOptions = "p:e:d:y:h:w:r:i:m:";
	const struct option longOptions[] = {
		{ "population", required_argument, nullptr, 'p' },
		{ "exposure-duration", required_argument, nullptr, 'e' },
		{ "infection-duration", required_argument, nullptr, 'd' },
		{ "dead-probability", required_argument, nullptr, 'r' },
		{ "initial-infected", required_argument, nullptr, 'i' },
		{ "initial-immune", required_argument, nullptr, 'm' },
		{ "proximity", required_argument, nullptr, 'y' },
		{ "world-height", required_argument, nullptr, 'h' },
		{ "world-width", required_argument, nullptr, 'w' },
		{ "help", no_argument, nullptr, ' ' },
		{ nullptr, 0, nullptr, 0 }
	};

	// Traitement des arguments
	int opt;
	while ((opt = getopt_long(argc, argv, shortOptions, longOptions,
				  nullptr)) != -1) {
		switch (opt) {
		case 'p':
			populationPercent = atoi(optarg);
			if (populationPercent < 0 || populationPercent > 100) {
				cerr << "Error: Population must be > 0% and <=100%.\n";
				return 1;
			}
			break;

		case 'e':
			exposureDuration = atoi(optarg);
			break;

		case 'd':
			infectionDuration = atoi(optarg);
			break;

		case 'r':
			deathProbability = atoi(optarg);
			break;
		case 'i':
			initialInfected = atoi(optarg);
			break;

		case 'm':
			initialImmune = atoi(optarg);
			break;

		case 'y':
			proximity = atoi(optarg);
			break;
		case 'h':
			worldHeight = atoi(optarg);
			break;

		case 'w':
			worldWidth = atoi(optarg);
			break;

		case '?':
		case ':':
		default:
			printUsage();
			return 1;
		}
	}
	cout << "Plague Simulator\n\n";
	cout << "worldHeight: " << worldHeight << "\n";
	cout << "worldWidth: " << worldWidth << "\n";
	// Réinitialiser les grilles en fonction des dimensions
	world = std::vector<vector<int> >(worldHeight,
					  vector<int>(worldWidth, EMPTY));
	exposureDurationMap = std::vector<vector<int> >(
		worldHeight, vector<int>(worldWidth, exposureDuration));
	infectionDurationMap = std::vector<vector<int> >(
		worldHeight, vector<int>(worldWidth, infectionDuration));

	cout << "Parameters :\n"
	     << "  Population : " << populationPercent << "%\n"
	     << "  World height : " << worldHeight << "\n"
	     << "  World Width : " << worldWidth << "\n\n";

	initializeGrid();
	initializeInfection(initialInfected);

	cout << "Initial world :\n";
	//displayWorld();
	cout << "Number of healty people: " << getNbHealty() << endl;
	cout << "Number of infected people: " << getNbInfected() << endl;
	cout << "Number of dead people: " << getNbDead() << endl;
	cout << "Number of empty cells: " << getNbEmpty() << endl;
	cout << "\nSimulation started\n";

	timespec timer;
	startTimer(timer);
	vector<vector<vector<int> > > stepsToApocalypse;
	int nb_turn = 0;

	vector<vector<int> > stepToApocalypse;

	//init timer
	while (getNbInfected() > 0) {
		updateWorld();
		nb_turn++;
		stepToApocalypse = world;
		stepsToApocalypse.push_back(stepToApocalypse);
		//displayWorld();
	}

	std::cout << "Simulation took " << stopTimer(timer) << " ms\n";
	cout << "Number of turns: " << nb_turn << endl;
	cout << "Number of healty people: " << getNbHealty() << endl;
	cout << "Number of infected people: " << getNbInfected() << endl;
	cout << "Number of dead people: " << getNbDead() << endl;
	cout << "Number of empty cells: " << getNbEmpty() << endl;
	createVideo(stepsToApocalypse, "plague.avi", 20, 10);
	return 0;
}
