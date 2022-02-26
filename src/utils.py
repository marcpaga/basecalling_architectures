"""Contains general utilities
"""
import zipfile
import numpy as np
import signal
from contextlib import contextmanager

def read_metadata(file_name):
    """Read the metadata of a npz file
    
    Args:
        filename (str): .npz file that we want to read the metadata from
        
    Returns:
        (list): with as many items as arrays in the file, each item in the list
        is filename (within the zip), shape, fortran order, dtype
    """
    zip_file=zipfile.ZipFile(file_name, mode='r')
    arr_names=zip_file.namelist()

    metadata=[]
    for arr_name in arr_names:
        fp=zip_file.open(arr_name,"r")
        version=np.lib.format.read_magic(fp)

        if version[0]==1:
            shape,fortran_order,dtype=np.lib.format.read_array_header_1_0(fp)
        elif version[0]==2:
            shape,fortran_order,dtype=np.lib.format.read_array_header_2_0(fp)
        else:
            print("File format not detected!")
        metadata.append((arr_name,shape,fortran_order,dtype))
        fp.close()
    zip_file.close()
    return metadata

def find_runs(x):
    """Find runs of consecutive items in an array."""

    # ensure array
    x = np.asanyarray(x)
    if x.ndim != 1:
        raise ValueError('only 1D array supported')
    n = x.shape[0]

    # handle empty array
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    else:
        # find run starts
        loc_run_start = np.empty(n, dtype=bool)
        loc_run_start[0] = True
        np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
        run_starts = np.nonzero(loc_run_start)[0]

        # find run values
        run_values = x[loc_run_start]

        # find run lengths
        run_lengths = np.diff(np.append(run_starts, n))

        return run_values, run_starts, run_lengths



class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    """Raise a TimeoutException if a function runs for too long

    Args:
        seconds (int): amount of max time the function can be run

    Example:
    try:
        with time_limit(10):
            my_func()
    except TimeoutException:
        do_something()
    """
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)