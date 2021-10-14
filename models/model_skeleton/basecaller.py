"""This script can be used to basecall a set of fast5 files given a model,
it will produce one (or more) fasta files with the basecalls
"""

import argparse

def main():
    pass


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast5-path", type=str, help='Path to fast5 files, it is searched recursively')
    parser.add_argument("--model", type=str, help='Path to the model configuration file')
    args = parser.parse_args()
    
    main()