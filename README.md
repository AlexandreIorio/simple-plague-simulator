# simple-plague-simulator

## Description

**Plague Simulator** is a program that simulates the spread of a plague within a population defined by various parameters. You can customize the simulation's behavior, view detailed results, and optionally generate a video showing the progression of the outbreak.

![gif](./docs/assets/img/plague.gif)

## Features

- Allows customization of parameters like population size, initial immunity, infection duration, and more.
- Provides detailed statistics on the progression of the plague.
- Optionally generates a video visualizing the outbreak's spread.

**Versions**

The main application can be built in three different ways:

1. Standard - Everything runs sequentially - No multithreading
2. OpenMP - Uses openmp to run code in parallel
3. Cuda - Compile using CUDA to get the best possible performance when using big grids

All three can be built using the main `Makefile`:

**Standard:**

```bash
make std
./build/std/app
```

**OpenMP:**

```bash
make omp
./build/openmp/app
```

**CUDA:**

```bash
make cuda
./build/cuda/app
```

## **Available Options**

| Short Option | Long Option            | Description                                            | Default Value |
| ------------ | ---------------------- | ------------------------------------------------------ | ------------- |
| `-p`         | `--population`         | Percentage of the population to simulate               | `50%`         |
| `-m`         | `--initial-immune`     | Number of initially immune people                      | `0`           |
| `-e`         | `--healthy-infection`  | Probability of a healthy people of getting infected    | `5`           |
| `-d`         | `--infection-duration` | Duration of infection before immunity or death (turns) | `10`          |
| `-r`         | `--dead-probability`   | Probability of death after infection                   | `10%`         |
| `-i`         | `--initial-infected`   | Number of initially infected individuals               | `1`           |
| `-y`         | `--proximity`          | Proximity required for infection                       | `2`           |
| `-h`         | `--world-height`       | Height of the simulation grid                          | `10`          |
| `-w`         | `--world-width`        | Width of the simulation grid                           | `10`          |
| `-r`         | `--rounds`             | Maximum number of rounds                               | `Unlimited`   |
| `-f`         |                        | Path to a file containing the simulation parameters    |               |
| `--help`     |                        | Display usage information                              |               |

##### Example:

```bash
./build/std/app -f parameters.txt
./build/openmp/app -f parameters.txt
./build/cuda/app -f parameters.txt
```

## Simulation Output

Upon execution, the program displays:

- Simulation parameters (population, immunity rate, grid dimensions, etc.).
- Initial statistics (number of healthy, infected, and immune individuals).
- Simulation progress and time taken.
- Final results: survivors, deaths, and remaining healthy population.

output example:

```bash
./build/openmp/app -f parameters.txt
-----------------------------------
         Plague Simulator
-----------------------------------
Runtime : OpenMP
------------------------------------
Parameters
------------------------------------
Population                  : 50%
World height                : 256
World Width                 : 256
World size                  : 65536
Proximity                   : 2
Infection duration          : 5 turns
Healthy infection probability:10 %
Immune infection probability: 0 %
Death probability           : 10%
Initial infected            : 1
Population immunized        : 0%

-----------------------------------
         Initialisation
-----------------------------------
Initializing World ...
Initialization Duration: 0.0117254 s
------------------------------------
Initial world :
------------------------------------
Number of healthy people  : 32767
Number of infected people : 1
Number of immunized people: 0

------------------------------------
Simulation
------------------------------------
Simulation started
...
------------------------------------
Saving Timeline
Timeline created
Initialization took       : 0.0117254 s
Simulation took           : 0.959415 s
Total Time                : 0.971141 s
Number of turns           : 397
Number of healthy people  : 3116
Number of immunized people: 26691
Number of survivor        : 29807
Number of dead people     : 2961
```
