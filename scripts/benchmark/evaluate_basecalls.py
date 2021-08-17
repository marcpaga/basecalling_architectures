"""This script will evaluate the basecalling performance, it produces two files:
    - A per read table file
    - A global report file
"""

import argparse

def main():
    pass


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast5-path", type=str, help='Path to fast5 files, it is searched recursively')
    args = parser.parse_args()
    
    main()