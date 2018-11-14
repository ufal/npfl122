#!/usr/bin/env python3
"""Embed compressed data into a Python module."""
__version__ = "1.0.0"
__author__  = "Milan Straka <straka@ufal.mff.cuni.cz>",

import argparse
import base64
import io
import sys
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument("--output", default="embedded_data.py", type=str,
                    help="Name of output Python file with embedded data.")
parser.add_argument("path", type=str, nargs="+",
                    help="Path to files and directories to be embedded.")
args = parser.parse_args()

print("Compressing given paths...", file=sys.stderr, end="")
tar_data = io.BytesIO()
with tarfile.open(fileobj=tar_data, mode="w:xz") as tar_file:
    for path in args.path:
        tar_file.add(path)
print("done.", file=sys.stderr)

with open(args.output, "w") as output_file:
    print("""#!/usr/bin/env python3

def extract():
    import base64
    import io
    import tarfile
    data = """, base64.b85encode(tar_data.getbuffer()), """
    with io.BytesIO(base64.b85decode(data)) as tar_data:
        with tarfile.open(fileobj=tar_data, mode="r") as tar_file:
            tar_file.extractall()

if __name__ == "__main__":
    extract()""", file=output_file, sep="")

print("Output file `{}` with embedded data created.".format(args.output), file=sys.stderr)
