# simple-plague-simulator

## Description

**Plague Simulator** is a program that simulates the spread of a plague within a population defined by various parameters. You can customize the simulation's behavior, view detailed results, and generate a video showing the progression of the outbreak.

![gif](./docs/assets/img/plague.gif)

## Getting Started

You can build the three versions directly using the provided Makefile.

```bash
make
```

This will build the three versions of the program. If you prefer only building one of the versions you can build them individually.

```bash
make std # Build the standard version
make omp # Build the openmp version
make cuda # Build the cuda version
```

We can now generate a parameter file using the following command:

```bash
./plague-simulator-std -g parameters.txt
```

This will generate a file named `parameters.txt` with the default parameters.

You can now run each version of the program using this parameter file.

```bash
./plague-simulator-std -f parameters.txt
./plague-simulator-omp -f parameters.txt
./plague-simulator-cuda -f parameters.txt
```

You can optionally limit the number of rounds using the `-r` option. This is useful to compare the performance of the different versions.

```bash
./plague-simulator-std -f parameters.txt -r 100
```

## Features

- Allows customization of parameters like population size, initial immunity, infection duration, and more.
- Provides detailed statistics on the progression of the plague.
- Optionally generates a video visualizing the outbreak's spread.

**Versions**

The main application can be built in three different ways:

1. Standard - Everything runs sequentially - No multithreading
2. OpenMP - Uses openmp to run code in parallel
3. Cuda - Compile using CUDA to get the best possible performance when using big grids

## **Available Options**

```bash
./plague-simulator-std --help
Usage: ./plague-simulator-std [options]
Options:
  -f, --file      <file_path>          Parameter File
  -g, --generate  <output_file_path>   Generates a Parameter File
  -r, --rounds    <value>              Max Rounds (default: No limit)
      --help                           Display this information
```

## Parameters

The parameter file is a simple text file that contains the parameters of the simulation. It looks like this:

```txt
height 256
width 256
initial_infected 1
initial_immune 0
proximity 2
population_percentage 100
death_probability 1
infection_duration 5
healthy_infection_probability 2
immune_infection_probability 1
```

## Timeline

Running a simulation generate a `timeline.bin` file that stores the state of the simulation at each turn.

> [!NOTE]
> This file is stored using RLE encoding to save space and the program limits the size of the file to 500MB to avoid running out of space.

With this `timeline.bin`, you can:

1. Generate a video
2. Display the video in real-time
3. Export the details of the simulation

### Generate a Video

First, make sure you have `opencv` installed on your system. `h264` codec is used to compress the video.

```bash
make generate_video
./generate_video timeline.bin output.avi
```

You can now open the video using your favorite video player.

### Display the Simulation

If you don't want to generate a video, you can display the simulation in real-time.

For this, you need to have `SDL2` installed on your system.

```bash
make display_timeline
./display_timeline timeline.bin
```

### Export the Details of the Simulation

This script will export a CSV file with the details of the simulation.

```bash
make timeline_details
./timeline_details timeline.bin output.csv
```

The format of the CSV file is as follows:

```csv
round,healthy,infected,immune,dead,total
0,524287,1,0,0,524288
1,524287,1,0,0,524288
...
```

## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
