#ifndef PLAGUE_H
#define PLAGUE_H

#include <vector>
#include <ctime>
using namespace std;

class Plague {
    public:
	int worldHeight;
	int worldWidth;
	int populationPercent;
	int immunePercent;
	int exposureDuration;
	int deathProbability;
	int infectionDuration;
	int initialInfected;
	int proximity;
	vector<vector<int> > world;
	vector<vector<int> > exposureDurationMap;
	vector<vector<int> > infectionDurationMap;
};

///@brief Enum representing the different states of a cell in the simulation.
enum State { EMPTY = 0, HEALTHY, INFECTED, DEAD, IMMUNE };

///@brief Initializes the grid with random values based on population percentage.
void initializeGrid(Plague &p);

///@brief Retrieves the indices of all healthy people in the grid.
///@param healthyPeopleIndexes Vector to store the indices of healthy people.
void getHealty(Plague p, vector<vector<int> > &healthyPeopleIndexes);

///@brief Retrieves the indices of all infected people in the grid.
///@param infectedPeopleIndexes Vector to store the indices of infected people.
void getInfected(Plague p, vector<vector<int> > &infectedPeopleIndexes);

///@brief Gets the total number of infected people in the grid.
///@return The number of infected people.
int getNbInfected(Plague p);

///@brief Gets the total number of healthy people in the grid.
///@return The number of healthy people.
int getNbHealty(Plague p);

///@brief Gets the total number of immune people in the grid.
///@return The number of immune people.
int getNbImmune(Plague p);

///@brief Gets the total number of dead people in the grid.
///@return The number of dead people.
int getNbDead(Plague p);

///@brief Infects a specified number of healthy people randomly.
///@param nbInfected The number of people to infect.
void initializeInfection(Plague &p);

///@brief Initializes immune people in the grid based on the immune percentage.
void initializeImmune(Plague &p);

///@brief Retrieves the neighbors of a cell within the proximity range.
///@param i Row index of the cell.
///@param j Column index of the cell.
///@param neighbours Vector to store the neighbors' indices.
void getNeighbours(Plague p, int i, int j,
		   std::vector<std::vector<int> > &neighbours);

///@brief Updates the world grid based on the simulation rules.
void updateWorld(Plague &p);

#endif // PLAGUE_H
