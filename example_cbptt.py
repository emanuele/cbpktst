"""Example of mass-univariate cluster-based permutation t-test
(CBPtT). This should be equivalent to what FieldTrip provides. This
implementation follows the description in Groppe et
al. (Psychophysiology, 2011) p.1718.
"""

import numpy as np
from simulate_data import simulate_2d, plot_map2d
from scipy.stats import ttest_ind
from cbptt import cluster_based_permutation_t_test

if __name__ == '__main__':

    seed = 0
    np.random.seed(seed)
    verbose = True
    parallel = True
    save = True
    space_sparse = False

    print("Mass-univariate cluster-based permutation t-test (CBPtT)")
    p_value_threshold = 0.05
    iterations = 1000
    n_jobs = -1
    batch_size = 50

    print("Creating simulated data.")
    d = 20
    k = 10
    m = 200
    n = 200
    threshold_space = 1.5
    threshold_timesteps = 1
    XX, YY, proximity_matrix, coordinates = simulate_2d(d=d, k=k,
                                                        m=m, n=n,
                                                        threshold=threshold_space)

    clusters, clusters_over_threshold = cluster_based_permutation_t_test(XX, YY, coordinates, iterations, p_value_threshold, threshold_space, threshold_timesteps, parallel=parallel, n_jobs=n_jobs, batch_size=batch_size, verbose=verbose, space_sparse=space_sparse)

    t, p = ttest_ind(XX, YY, axis=0)
    plot_map2d(k, coordinates, t.mean(1))

    print("Keeping only significant clusters and producing space values.")
    n_units = k * k
    thresholded_t_map = np.zeros(n_units)
    for i, cot in enumerate(clusters_over_threshold):
        for unit_ts in clusters[cot]:
            unit_s = unit_ts % n_units
            timestep = unit_ts // n_units
            thresholded_t_map[unit_s] = t[unit_s, timestep]

    plot_map2d(k, coordinates, thresholded_t_map)
