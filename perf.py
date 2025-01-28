import subprocess

TARGETS = ["std", "openmp", "cuda"]


def main():

    for target in TARGETS:
        p = subprocess.Popen(
            ["build", target, "app", "-f", "parameters.txt", "-r", "100"],
            stdout=subprocess.PIPE,
        )

        stdout, _ = p.communicate()

        time_lines = stdout.decode().splitlines()[-7:-10]
        times = [line.split(":")[1].replace("s", "").strip() for line in time_lines]
        init, sim, total = times

        print(init, sim, total)


if __name__ == "__main__":
    main()
