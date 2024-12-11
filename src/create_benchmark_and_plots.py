import numpy as np
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from geostatbench import BayesInvBench, parula_map

benchmark = BayesInvBench('config/config.yaml')
# add directory of data
# set reference path
benchmark.set_reference_path()
# set reference chain file paths
benchmark.set_reference_chain_file_paths()
benchmark.load_replication_piezometric_head()
benchmark.prepare_reference_data()
benchmark.plot_ref_mean_field()
benchmark.get_benchmark_table() # Storing it in benchmark
figS0_1000,tableS0_1000 = benchmark.make_comparison_plot()

# figS4_1000,tableS4_1000 = benchmark.make_comparison_plot()
# figS4_1000