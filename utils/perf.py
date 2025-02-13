import subprocess

TARGETS = ["std", "omp", "cuda"]

WIDTH = [int(2**i) for i in range(2, 14)]
HEIGHT = [int(2**i) for i in range(2, 14)]
ROUNDS = 100

BENCHMARK_FILE_PATH = "benchmark_parameters.txt"
PARAMETERS_FILE_CONTENT = """initial_infected 1
initial_immune 0
proximity 2
population_percentage 50
death_probability 10
infection_duration 5
healthy_infection_probability 10
immune_infection_probability 1
"""

BENCHMARK_RESULT_FILE = "benchmark_result.csv"


def main():

    csv_lines = []
    for width, height in zip(WIDTH, HEIGHT):
        with open(BENCHMARK_FILE_PATH, "w") as f:
            f.write(PARAMETERS_FILE_CONTENT)
            f.write(f"width {width}\nheight {height}\n")

        for target in TARGETS:
            p = subprocess.Popen(
                [
                    f"./plague-simulator-{target}",
                    "-f",
                    BENCHMARK_FILE_PATH,
                    "-r",
                    str(ROUNDS),
                ],
                stdout=subprocess.PIPE,
            )

            stdout, _ = p.communicate()

            print(f"{target} - ({width} {height})")
            lines = stdout.decode().splitlines()
            time_lines = lines[-9:-6]
            times = [line.split(":")[1].replace("s", "").strip() for line in time_lines]
            rounds = lines[-6].split(":")[1].strip()
            init, sim, total = times
            time_per_round = float(sim) / float(rounds)
            params = (target, width, height, rounds, init, sim, time_per_round, total)

            csv_lines.append(",".join(str(param) for param in params) + "\n")

    with open(BENCHMARK_RESULT_FILE, "w") as f:
        f.write(
            "target,width,height,rounds,time_init,time_simulation,time_per_round,time_total\n"
        )
        f.writelines(csv_lines)


if __name__ == "__main__":
    main()
