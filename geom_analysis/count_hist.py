import sys
import numpy as np

nn_file = sys.argv[1]
cutoff=3.0
print('cutoff = ',cutoff)

nn_dist = np.loadtxt(nn_file)
nn_belowcut = nn_dist[nn_dist <= cutoff]
print(np.count_nonzero(nn_belowcut))
# print(nn_dist[0])
