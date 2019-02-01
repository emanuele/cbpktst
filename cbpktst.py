"""Cluster-Based Permutation Kernel Two-Sample Test (CBPKTST).

See Olivett et al. (2014).
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from kernel_two_sample_test.kernel_two_sample_test import (MMD2u,
                                                           compute_null_distribution,
                                                           compute_null_distribution_given_permutations)
from joblib import Parallel, delayed
from scipy.sparse import issparse
from networkx import (from_scipy_sparse_matrix, from_numpy_matrix,
                      connected_components)
from sys import stdout


def precompute_gaussian_kernels(XX, YY, verbose=False):
    """For each unit, precompute Gaussian kernel between the trials of
    the two samples XX and YY. Estimate each sigma2 parameter as median
    distance between the trials of each sample.
    """
    if verbose:
        print("Pre-computing the kernel matrix for each unit.")

    n_units = XX.shape[1]  # or YY.shape[1]
    Ks = []  # here we store all the kernel matrices
    sigma2s = np.zeros(n_units)  # here we store all the sigma2s, one per unit
    m = XX.shape[0]
    n = YY.shape[0]
    for i in range(n_units):
        if verbose: print("Unit %s" % i),
        X = XX[:,i,:].copy()
        Y = YY[:,i,:].copy()
        if verbose: print("Computing Gaussian kernel."),
        dm = pairwise_distances(np.vstack([X, Y]), metric='sqeuclidean')
        # Heuristic: sigma2 is the median value among all pairwise
        # distances between X and Y. Note: should we use just
        # dm[:m,m:] or all dm?
        sigma2 = np.median(dm[:m,m:]) 
        sigma2s[i] = sigma2
        if verbose: print("sigma2 = %s" % sigma2)
        K = np.exp(-dm / sigma2)
        Ks.append(K)

    return Ks, sigma2s


def compute_mmd2u_and_null_distributions(Ks, m, n, iterations=1000,
                                         seed=0, parallel=True,
                                         permutation=None, n_jobs=-1,
                                         verbose=False):
    """Compute MMD2u statistic and its null-distribution for each unit
    from kernel matrices Ks. Each null-distributions is approximated
    with the given number of iterations. Parallel (multiprocess, with
    n_jobs processes) computation is available. Note: n_jobs=-1 means
    'use all available cores'. Precomputed permutations (array of size
    iterations x (m+n)) can be used instead of randomly generated
    ones to enforce reproducibility and keep the desired permutation
    schema for each kernel/unit. This is important during parallel
    computation.
    """
    n_units = len(Ks)
    unit_statistic = np.zeros(n_units)
    unit_statistic_permutation = np.zeros((n_units, iterations))

    print("Computing MMD2u for each unit.")
    for i, K in enumerate(Ks):
        mmd2u = MMD2u(K, m, n)
        unit_statistic[i] = mmd2u

    print("Computing MMD2u's null-distribution, for each unit.")
    if not parallel:
        for i, K in enumerate(Ks):
            if permutation is None:
                 # NOTE: IT IS FUNDAMENTAL THAT THE SAME SEED IS USED
                 # FOR EACH UNIT!
                mmd2u_null = compute_null_distribution(K, m, n,
                                                       iterations=iterations,
                                                       verbose=verbose,
                                                       random_state=seed,
                                                       marker_interval=100)
            else:
                mmd2u_null = compute_null_distribution_given_permutations(K, m, n,
                                                                          permutation,
                                                                          iterations=iterations)
            
            unit_statistic_permutation[i, :] = mmd2u_null
    else:
        print("Parallel computation!")
        if permutation is None:
            # NOTE: IT IS FUNDAMENTAL THAT THE SAME SEED IS USED FOR EACH UNIT!
            results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(compute_null_distribution)(K, m, n, iterations=iterations, verbose=False, random_state=seed) for K in Ks)
        else:
            results = Parallel(n_jobs=n_jobs, verbose=10)(delayed(compute_null_distribution_given_permutations)(K, m, n, permutation, iterations=iterations) for K in Ks)
            
        unit_statistic_permutation = np.vstack(results)

    return unit_statistic, unit_statistic_permutation


def compute_clusters_statistic(test_statistic, proximity_matrix,
                               verbose=False):
    """Given a test statistic for each unit and a boolean proximity
    matrix among units, compute the cluster statistic using the
    connected components graph algorithm. It works for sparse
    proximity matrices as well.

    Returns the clusters and their associated cluster statistic.
    """
    # Build a graph from the proximity matrix:
    if issparse(proximity_matrix):
        graph = from_scipy_sparse_matrix(proximity_matrix)
    else:
        graph = from_numpy_matrix(proximity_matrix)

    # Compute connected components and transform in list of lists:
    clusters = [list(cluster) for cluster in connected_components(graph)]
    if verbose:
        print("Nr. of clusters: %s. Clusters sizes: %s" % (len(clusters), np.array([len(cl) for cl in clusters])))

    # Compute the cluster statistic:
    cluster_statistic = np.zeros(len(clusters))
    for i, cluster in enumerate(clusters):
        cluster_statistic[i] = test_statistic[cluster].sum()

    # final cleanup to prepare easy-to-use results:
    idx = np.argsort(cluster_statistic)[::-1]
    clusters = np.array([np.array(cl, dtype=np.int) for cl in
                         clusters], dtype=np.object)[idx]
    # THIS FIXES A NUMPY BUG (OR FEATURE?)
    if clusters[0].dtype == np.object:
        # The bug: it seems not possible to create ndarray of type
        # np.object from arrays all of the *same* lenght and desired
        # dtype, i.e. dtype!=np.object. In this case the desired dtype
        # is automatically changed into np.object. Example:
        # array([array([1], dtype=int)], dtype=object)
        clusters = clusters.astype(np.int)

    cluster_statistic = cluster_statistic[idx]
    return clusters, cluster_statistic


def compute_pvalues_from_permutations(statistic, statistic_permutation):
    """Efficiently compute p-value(s) of statistic given permuted
    statistics.

    Note: statistic can be a vector and statistic_permutation can be a
    matrix units x permutations.
    """
    statistic_permutation = np.atleast_2d(statistic_permutation)
    iterations = statistic_permutation.shape[1]
    p_value = (statistic_permutation.T >= statistic).sum(0).astype(np.float) / iterations
    return p_value


def compute_pvalues_of_permutations(statistic_permutation):
    """Given permutations of a statistic, compute the p-value of each
    permutation.

    Note: tatistic_permutation can be a matrix units x permutations.
    """
    statistic_permutation = np.atleast_2d(statistic_permutation)
    iterations = statistic_permutation.shape[1]
    p_value_permutation = (iterations - np.argsort(np.argsort(statistic_permutation, axis=1), axis=1)).astype(np.float) / iterations # argsort(argsor(x)) given the rankings of x in the same order. Example: a=[60,35,70,10,20] , then argsort(argsort(a)) gives array([3, 2, 4, 0, 1])
    return p_value_permutation


def compute_statistic_threshold(statistic_permutation, p_value_threshold):
    """Compute the threshold of a statistic value given permutations
    and p_value_threshold.
    
    Note: tatistic_permutation can be a matrix units x permutations.
    """
    statistic_permutation = np.atleast_2d(statistic_permutation)
    iterations = statistic_permutation.shape[1]
    statistic_threshold = np.sort(statistic_permutation, axis=1)[:, np.int((1.0-p_value_threshold)*iterations) - 1]
    return statistic_threshold.squeeze()


def compute_homogeneous_statistics(unit_statistic,
                                   unit_statistic_permutation,
                                   p_value_threshold,
                                   homogeneous_statistic='normalized MMD2u',
                                   verbose=True):
    """Compute p_values from permutations and create homogeneous statistics.
    """
    # Compute p-values for each unit    
    print("Homogeneous statistic: %s" % homogeneous_statistic)
    print("Computing MMD2u thresholds for each unit with p-value=%f" % p_value_threshold)
    mmd2us_threshold = compute_statistic_threshold(unit_statistic_permutation, p_value_threshold)
    print("Computing actual p-values at each unit on the original (unpermuted) data")
    p_value = compute_pvalues_from_permutations(unit_statistic, unit_statistic_permutation)
    print("Computing the p-value of each permutation of each unit.")
    p_value_permutation = compute_pvalues_of_permutations(unit_statistic_permutation)

    # Here we try to massage the unit statistic so that it becomes
    # homogeneous across different units, to compute the cluster
    # statistic later on
    if homogeneous_statistic == '1-p_value': # Here we use (1-p_value) instead of the MMD2u statistic : this is perfectly homogeneous across units because the p_value is uniformly distributed, by definition
        unit_statistic_permutation_homogeneous = 1.0 - p_value_permutation
        unit_statistic_homogeneous = 1.0 - p_value
    elif homogeneous_statistic == 'normalized MMD2u': # Here we use a z-score of MMD2u, which is good if its distribution normal or approximately normal
        mmd2us_mean = unit_statistic_permutation.mean(1)
        mmd2us_std = unit_statistic_permutation.std(1)
        unit_statistic_permutation_homogeneous = np.nan_to_num((unit_statistic_permutation - mmd2us_mean[:,None]) / mmd2us_std[:,None])
        unit_statistic_homogeneous = np.nan_to_num((unit_statistic - mmd2us_mean) / mmd2us_std)
    elif homogeneous_statistic == 'unit_statistic': # Here we use the unit statistic assuming that it is homogeneous across units (this is not much true)
        unit_statistic_permutation_homogeneous = unit_statistic_permutation
        unit_statistic_homogeneous = unit_statistic
    elif homogeneous_statistic == 'p_value': # Here we use p_value instead of the MMD2u statistic : this is perfectly homogeneous across units because the p_value is uniformly distributed, by definition
        unit_statistic_permutation_homogeneous = p_value_permutation
        unit_statistic_homogeneous = p_value
    else:
        raise Exception

    return p_value, p_value_permutation, unit_statistic_homogeneous, unit_statistic_permutation_homogeneous


def cluster_based_permutation_test(unit_statistic,
                                   unit_statistic_permutation,
                                   proximity_matrix,
                                   p_value_threshold=0.05,
                                   homogeneous_statistic='normalized MMD2u',
                                   verbose=True):
    """This is the cluster-based permutation test of CBPKTST, where
    the MMD2u permutations at each unit are re-used in order to
    compute the max_cluster_statistic.
    """
    p_value, p_value_permutation, unit_statistic_homogeneous, unit_statistic_permutation_homogeneous = compute_homogeneous_statistics(unit_statistic, unit_statistic_permutation, p_value_threshold, homogeneous_statistic=homogeneous_statistic, verbose=verbose)
    
    # Compute clusters and max_cluster_statistic on permuted data

    unit_significant = p_value <= p_value_threshold
    unit_significant_permutation = p_value_permutation <= p_value_threshold
    iterations = unit_statistic_permutation.shape[1]

    print("For each permutation compute the max cluster statistic.")
    max_cluster_statistic = np.zeros(iterations)
    for i in range(iterations):
        max_cluster_statistic[i] = 0.0
        if unit_significant_permutation[:,i].sum() > 0:
            idx = np.where(unit_significant_permutation[:,i])[0]
            # BEWARE! If you don't use where() in the previous line
            # but stick with boolean indices, then the next slicing
            # fails when proximity_matrix is sparse. See:
            # http://stackoverflow.com/questions/6408385/index-a-scipy-sparse-matrix-with-an-array-of-booleans
            pm_permutation = proximity_matrix[idx][:,idx]
            print("%d" % i),
            stdout.flush()
            cluster_permutation, cluster_statistic_permutation = compute_clusters_statistic(unit_statistic_permutation_homogeneous[idx,i], pm_permutation, verbose=verbose)
            # Mapping back clusters to original ids:
            cluster_permutation = np.array([idx[cp] for cp in cluster_permutation])
            max_cluster_statistic[i] = cluster_statistic_permutation.max()

    print("Computing the null-distribution of the max cluster statistic.")
    max_cluster_statistic_threshold = compute_statistic_threshold(max_cluster_statistic, p_value_threshold)
    print("Max cluster statistic threshold (p-value=%s) = %s" % (p_value_threshold, max_cluster_statistic_threshold))

    # Compute clusters and max_cluster_statistic on the original
    # (unpermuted) data

    print("")
    print("Computing significant clusters on unpermuted data.")
    idx = np.where(unit_significant)[0] # no boolean idx for sparse matrices!
    cluster_significant = []
    if len(idx) > 0:
        pm = proximity_matrix[idx][:,idx]
        cluster, cluster_statistic = compute_clusters_statistic(unit_statistic_homogeneous[idx], pm, verbose=True)
        # Mapping back clusters to original ids:
        cluster = np.array([idx[c] for c in cluster])
        print("Cluster statistic: %s" % cluster_statistic)
        p_value_cluster = compute_pvalues_from_permutations(cluster_statistic, max_cluster_statistic)
        print "p_value_cluster:", p_value_cluster
        cluster_significant = cluster[p_value_cluster <= p_value_threshold]
        print("%d significant clusters left" % len(cluster_significant))
    else:
        print("No clusters in unpermuted data!")
        cluster = np.array([])
        cluster_statistic = np.array([])
        p_value_cluster = np.array([])

    print("Zeroing all unit statistic (homogeneous too) related non-significant clusters.")
    unit_statistic_significant = np.zeros(unit_statistic.size)
    unit_statistic_homogeneous_significant = np.zeros(unit_statistic.size)
    for cs in cluster_significant:
        unit_statistic_significant[cs] = unit_statistic[cs]
        unit_statistic_homogeneous_significant[cs] = unit_statistic_homogeneous[cs]

    return cluster, cluster_statistic, p_value_cluster, p_value_threshold, max_cluster_statistic, unit_statistic_homogeneous
