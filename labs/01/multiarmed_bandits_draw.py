#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt
import numpy as np

import multiarmed_bandits

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Experiment to draw, or `all`")
    args = parser.parse_args()

    experiments = {
        "greedy-avg": {"epsilon": [1/64, 1/32, 1/16, 1/8, 1/4]},
        "greedy-lr=.15": {"alpha": 0.15, "epsilon": [1/64, 1/32, 1/16, 1/8, 1/4]},
        "greedy-lr=.15-init=1": {"alpha": 0.15, "initial": 1, "epsilon": [1/128, 1/64, 1/32, 1/16]},
        "ucb": {"c": [1/4, 1/2, 1, 2, 4]},
        "gradient": {"alpha": [1/16, 1/8, 1/4, 1/2]},
    }

    for e in sorted(experiments) if args.experiment == "all" else [args.experiment]:
        args = multiarmed_bandits.parser.parse_args([])
        args.mode = e.split("-")[0]

        for option, value in experiments[e].items():
            if isinstance(value, list):
                x_name, x_values = option, value
            else:
                setattr(args, option, value)

        results = []
        for x_value in x_values:
            setattr(args, x_name, x_value)
            results.append(multiarmed_bandits.main(args))
        means, stds = map(np.array, zip(*results))

        plt.plot(x_values, means, label=e)
        plt.fill_between(x_values, means - stds, means + stds, alpha=0.5)
    plt.xscale("log", basex=2)
    plt.xlabel("Epsilon/C/Alpha")
    plt.ylabel("Average Return")
    plt.legend(loc="lower right")
    plt.show()
