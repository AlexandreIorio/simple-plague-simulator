import subprocess

TARGETS = ["std", "openmp", "cuda"]

WIDTH = [int(2**i) for i in range(2, 14)]
HEIGHT = [int(2**i) for i in range(2, 14)]
ROUNDS = 100

BENCHMARK_FILE_PATH = "benchmark_parameters.txt"
PARAMETERS_FILE_CONTENT = """population 50
healthy_infection_probability 5
immune_infection_probability 0
death_probability 10
initial_infected 1
initial_immune 0
proximity 2"""

BENCHMARK_RESULT_FILE = "benchmark_result.csv"


def main():

    with open(BENCHMARK_FILE_PATH, "w") as f:
        f.write(PARAMETERS_FILE_CONTENT)

    csv_lines = []
    for target, width, height in zip(TARGETS, WIDTH, HEIGHT):
        p = subprocess.Popen(
            [
                f"build/{target}/app",
                "-f",
                BENCHMARK_FILE_PATH,
                "-r",
                str(ROUNDS),
                "-w",
                str(width),
                "-h",
                str(height),
            ],
            stdout=subprocess.PIPE,
        )

        stdout, _ = p.communicate()

        lines = stdout.decode().splitlines()
        time_lines = lines[-9:-6]
        times = [line.split(":")[1].replace("s", "").strip() for line in time_lines]
        rounds = lines[-6].split(":")[1].strip()
        init, sim, total = times
        params = (target, width, height, rounds, init, sim, total)
        csv_lines.append(",".join(str(param) for param in params))

    with open(BENCHMARK_RESULT_FILE, "w") as f:
        f.write("target, width, height, rounds, time_init, time_simulation, time_total")
        f.writelines(csv_lines)


if __name__ == "__main__":
    main()
