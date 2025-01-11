# simple-plague-simulator


## Description

**Plague Simulator** is a program that simulates the spread of a plague within a population defined by various parameters. You can customize the simulation's behavior, view detailed results, and optionally generate a video showing the progression of the outbreak.


## Features

- Allows customization of parameters like population size, initial immunity, infection duration, and more.
- Provides detailed statistics on the progression of the plague.
- Optionally generates a video visualizing the outbreak's spread.



## Requirements

- **C++ Compiler** (compatible with C++17 or later).
- **Dependencies**:
  - `plague.h` and `video_utils.h`.
  - OpenCV to create the video.


## Usage

Compile and run the program, specifying optional parameters to customize the simulation.

### **Compilation**
```bash
make
```

### **Execution**
```bash
./PlagueSimulator [options]
```

### **Available Options**
| Short Option | Long Option          | Description                                             | Default Value   |
|--------------|----------------------|---------------------------------------------------------|-----------------|
| `-p`         | `--population`       | Percentage of the population to simulate               | `50%`           |
| `-m`         | `--initial-immune`   | Percentage of initially immune population              | `0%`            |
| `-e`         | `--exposure-duration`| Duration of exposure before infection (turns)          | `5`             |
| `-d`         | `--infection-duration` | Duration of infection before immunity or death (turns) | `10`            |
| `-r`         | `--dead-probability` | Probability of death after infection                   | `10%`           |
| `-i`         | `--initial-infected` | Number of initially infected individuals               | `1`             |
| `-y`         | `--proximity`        | Proximity required for infection                       | `2`             |
| `-h`         | `--world-height`     | Height of the simulation grid                          | `10`            |
| `-w`         | `--world-width`      | Width of the simulation grid                           | `10`            |
| `-v`         | `--video`            | Generate a video showing the simulation                | Disabled        |
| `--help`     |                      | Display usage information                               |                 |

##### Example:
```bash
./PlagueSimulator -p 80 -m 10 -e 3 -d 8 -r 15 -i 5 -y 3 -h 20 -w 20 -v
```


## Simulation Output

Upon execution, the program displays:

- Simulation parameters (population, immunity rate, grid dimensions, etc.).
- Initial statistics (number of healthy, infected, and immune individuals).
- Simulation progress and time taken.
- Final results: survivors, deaths, and remaining healthy population.

If the `-v` flag is set, the program generates a video file (`plague.avi`) showing the simulation.

output example:
```bash
❯ ./PlagueSimulator -p 80 -m 10 -e 3 -d 8 -r 15 -i 5 -y 3 -h 20 -w 20

-----------------------------------
         Plague Simulator
-----------------------------------
------------------------------------
Parameters :
------------------------------------
Population           : 80%  
Population immunized : 10%  
World height         : 20
World Width          : 20
World size           : 400
Proximity            : 3
Exposure duration    : 3 turns
Infection duration   : 8 turns
Death probability    : 15%
Initial infected     : 5

------------------------------------
Initial world :
------------------------------------
Number of healty people   : 283
Number of infected people : 5
Number of immunized people: 32

------------------------------------
Simulation
------------------------------------
Simulation started
Simulation took           : 0.0236509 ms
Number of turns           : 21
Number of healty people   : 0
Number of immunized people: 278
Number of survivor        : 246
Number of dead people     : 42
```

width video flag:
```bash
❯ ./PlagueSimulator -p 45 -m 10 -e 3 -d 8 -r 15 -i 5 -y 3 -h 128 -w 128 -v

-----------------------------------
         Plague Simulator
-----------------------------------
------------------------------------
Parameters :
------------------------------------
Population           : 45%

...
...

------------------------------------
Simulation
------------------------------------
Simulation started
Simulation took           : 12.1546 s
Number of turns           : 109
Number of healty people   : 0
Number of immunized people: 6441
Number of survivor        : 5704
Number of dead people     : 931

------------------------------------
Creating video
------------------------------------
Creating...
Video created
```
![gif](./docs/assets/img/plague.gif)

