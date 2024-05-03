# import os
# import numpy as np
#
# def loadFile(fname):
#     # get the raw data
#     try:
#         f = open(os.path.join(fname))
#         table = f.read()
#         f.close()
#     except:
#         print("Opening files failed!")
#         exit()
#
#     lines = table.split('\n')
#     header = lines[0].split(',')
#
#     # drop the header line
#     lines = lines[1:]
#     print("header:", header)
#     print("date 1:", lines[0])
#     print("total sample points:", len(lines))
#     header = header[1:]  # drop 'Date' column
#
#     # convert the raw data to floating point numbers
#     raw_data = np.zeros((len(lines), len(header)))
#     for i, line in enumerate(lines):
#         values = [float(x) for x in line.split(',')[1:]]
#         raw_data[i, :] = values
#
#     return raw_data

import os
import numpy as np

def loadFile(fname):
    # get the raw data
    try:
        with open(os.path.join(fname)) as f:
            table = f.read()
    except:
        print("Opening files failed!")
        exit()

    lines = table.split('\n')
    header = lines[0].split(',')

    # drop the header line
    lines = lines[1:]
    print("header:", header)
    print("date 1:", lines[0])
    print("total sample points:", len(lines))
    header = header[1:]  # drop 'Date' column

    # convert the raw data to floating point numbers
    raw_data = np.zeros((len(lines), len(header)))
    for i, line in enumerate(lines):
        values = line.split(',')[1:]
        # Handle potential double quotes in values
        values = [float(x.strip('"')) if '"' in x else float(x) for x in values]

        # Check if the number of values matches the expected number of columns
        if len(values) != len(header):
            print(f"Skipping line {i+1} due to mismatch in number of columns.")
            continue

        raw_data[i, :] = values

    # Trim raw_data in case some rows were skipped
    raw_data = raw_data[:i+1, :]

    return raw_data