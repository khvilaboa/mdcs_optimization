from mercury.allocation_manager import AllocationManager
import pandas as pd
import numpy as np
import sys
import time
import pickle


def calc_time(func, *args, **kwargs):
    time_start = time.time()
    res = func(*args, **kwargs)
    time_end = time.time()
    return (time_end-time_start), res


df = pd.read_csv(sys.argv[1], delimiter=",")
t, alloc = calc_time(AllocationManager, df)
print("AllocationManager created (%d sec.)" % t)

alloc.allocate_aps()
alloc.allocate_edcs()
alloc.plot_scenario()
alloc.plot_grid()
alloc.store_scenario("data/")
