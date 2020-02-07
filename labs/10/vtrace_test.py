#!/usr/bin/env python3
import argparse
import importlib
import pickle

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("module", type=str, help="Module to test")
    parser.add_argument("--vtrace_data", default="vtrace_test.pickle", type=str, help="Data to test")
    args = parser.parse_args()

    # Load the module with the vtrace implementation
    implementation = importlib.import_module(args.module).Network.vtrace

    # Load the data
    with open(args.vtrace_data, "rb") as vtrace_data_file:
        vtrace_data = pickle.load(vtrace_data_file)

    # Test the implementation
    for inputs, outputs in zip(vtrace_data["inputs"], vtrace_data["outputs"]):
        results = implementation(*inputs)
        for i in range(2):
            if not np.allclose(results[i], outputs[i]):
                print("Different output[{}] for n={}, clip_c={}, clip_rho={}, gamma={},\ngot {}, expected {}".format(
                    i, inputs[0].n, inputs[0].clip_c, inputs[0].clip_rho, inputs[0].gamma, results[i], outputs[i]))
                exit(1)
    print("All OK")
