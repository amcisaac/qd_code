#!/bin/bash

# program to convert QChem scratch (binary) files into numpy binary files
# only does 53.0 and 320.0, for use with PDOS

echo "Converting 53.0 to ASCII"
hexdump -v -e '1/8 "% .16e " "\n"' 53.0 > 53.txt # convert MO coeff file to ascii
echo "Done with 53.0. Converting 320.0 to ASCII"
hexdump -v -e '1/8 "% .16e " "\n"' 320.0 > 320.txt # convert S matrix to ascii
echo "Done with 320.0. Converting txt files to numpy files"
python3 `dirname "$0"`/hex_to_npy.py nbas.txt 53.txt 320.txt # convert ascii to npy
