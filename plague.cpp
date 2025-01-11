
#include <vector>
#include <getopt.h>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <random>
#include "plague.h"

using namespace std;

void initializeGrid(Plague &p)
{
	srand(time(0));
	int maxPeople =
		p.worldHeight * p.worldWidth * p.populationPercent / 100;
	int people = 0;

	while (people < maxPeople) {
		for (int i = 0; i < p.worldHeight; i++) {
			for (int j = 0; j < p.worldWidth; j++) {
				if (p.world[i][j] != EMPTY) {
					continue;
				};
				if (people >= maxPeople) {
					break;
				}

				if (rand() % 100 < p.populationPercent) {
					p.world[i][j] = HEALTHY;
					people++;
				} else {
					p.world[i][j] = EMPTY;
				}
			}
		}
	}
}

int getNbInfected(Plague p)
{
	int nbInfected = 0;
	for (int i = 0; i < p.worldHeight; i++) {
		for (int j = 0; j < p.worldWidth; j++) {
			if (p.world[i][j] == INFECTED) {
				nbInfected++;
			}
		}
	}
	return nbInfected;
}

int getNbHealty(Plague p)
{
	int nbHealty = 0;
	for (int i = 0; i < p.worldHeight; i++) {
		for (int j = 0; j < p.worldWidth; j++) {
			if (p.world[i][j] == HEALTHY) {
				nbHealty++;
			}
		}
	}
	return nbHealty;
}

int getNbImmune(Plague p)
{
	int nbImmune = 0;
	for (int i = 0; i < p.worldHeight; i++) {
		for (int j = 0; j < p.worldWidth; j++) {
			if (p.world[i][j] == IMMUNE) {
				nbImmune++;
			}
		}
	}
	return nbImmune;
}

int getNbDead(Plague p)
{
	int nbDead = 0;
	for (int i = 0; i < p.worldHeight; i++) {
		for (int j = 0; j < p.worldWidth; j++) {
			if (p.world[i][j] == DEAD) {
				nbDead++;
			}
		}
	}
	return nbDead;
}

void initializeInfection(Plague &p)
{
	std::vector<std::pair<int, int> > cells;
	for (int i = 0; i < p.worldHeight; ++i) {
		for (int j = 0; j < p.worldWidth; ++j) {
			cells.emplace_back(i, j);
		}
	}

	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(cells.begin(), cells.end(), g);

	int infected = 0;
	for (const auto &cell : cells) {
		int i = cell.first;
		int j = cell.second;

		if (p.world[i][j] == HEALTHY) {
			p.world[i][j] = INFECTED;
			++infected;

			if (infected >= p.initialInfected) {
				break;
			}
		}
	}
}

void initializeImmune(Plague &p)
{
	srand(time(0));
	int maxPeople = getNbHealty(p) * p.immunePercent / 100;
	vector<pair<int, int> > healtyPeopleIndexes;

	for (int i = 0; i < p.worldHeight; i++) {
		for (int j = 0; j < p.worldWidth; j++) {
			if (p.world[i][j] == HEALTHY) {
				healtyPeopleIndexes.push_back({ i, j });
			}
		}
	}

	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(healtyPeopleIndexes.begin(), healtyPeopleIndexes.end(), g);

	for (int i = 0; i < maxPeople; i++) {
		p.world[healtyPeopleIndexes[i].first]
		       [healtyPeopleIndexes[i].second] = IMMUNE;
	}
}

void getNeighbours(Plague p, int i, int j, vector<vector<int> > &neighbours)
{
	neighbours.clear();
	for (int dx = -p.proximity; dx <= p.proximity; ++dx) {
		for (int dy = -p.proximity; dy <= p.proximity; ++dy) {
			if (dx == 0 && dy == 0) {
				continue;
			}

			int ni = i + dx;
			int nj = j + dy;

			// Vérification des limites de la grille
			if (ni >= 0 && ni < p.worldHeight && nj >= 0 &&
			    nj < p.worldWidth) {
				if (p.world[ni][nj] != EMPTY) {
					neighbours.push_back({ ni, nj });
				}
			}
		}
	}
}

void updateWorld(Plague &p)
{
	vector<vector<int> > tmpWorld(p.worldHeight,
				      vector<int>(p.worldWidth, EMPTY));
	for (int i = 0; i < p.worldHeight; i++) {
		for (int j = 0; j < p.worldWidth; j++) {
			tmpWorld[i][j] = p.world[i][j];
		}
	}

	vector<vector<int> > neighbours;

	for (int i = 0; i < p.worldHeight; i++) {
		for (int j = 0; j < p.worldWidth; j++) {
			switch (p.world[i][j]) {
			case HEALTHY:
				getNeighbours(p, i, j, neighbours);
				for (auto &neighbour : neighbours) {
					if (p.world[neighbour[0]]
						   [neighbour[1]] == INFECTED) {
						if (static_cast<float>(rand() %
								       100) /
							    100 <
						    p.healthyInfectionProbability) {
							tmpWorld[i][j] =
								INFECTED;
						}
						break;
					}
				}
				break;
			case INFECTED:
				if (p.infectionDurationMap[i][j] == 0) {
					if (static_cast<float>(rand() % 100) /
						    100 <
					    p.deathProbability) {
						tmpWorld[i][j] = DEAD;
					} else {
						tmpWorld[i][j] = IMMUNE;
						p.infectionDurationMap[i][j] =
							p.infectionDuration;
					}
				} else {
					p.infectionDurationMap[i][j]--;
				}

				break;
			case IMMUNE:
				getNeighbours(p, i, j, neighbours);
				for (auto &neighbour : neighbours) {
					if (p.world[neighbour[0]]
						   [neighbour[1]] == INFECTED) {
						if (static_cast<float>(rand() %
								       100) /
							    100 <
						    p.immuneInfectionProbability) {
							tmpWorld[i][j] =
								INFECTED;
						}
						break;
					}
				}
				break;
			case EMPTY:
			case DEAD:
				break;
			}
		}
	}

	for (int i = 0; i < p.worldHeight; i++) {
		for (int j = 0; j < p.worldWidth; j++) {
			p.world[i][j] = tmpWorld[i][j];
		}
	}
}
