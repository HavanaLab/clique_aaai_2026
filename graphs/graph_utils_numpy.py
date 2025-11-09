import numpy as np
from scipy import stats
from datetime import datetime



def get_top_k_degrees(adj_matrix, k):
    """
    Get the top k degrees from an adjacency matrix.

    Parameters:
    adj_matrix (numpy.ndarray): Adjacency matrix representing the graph.
    k (int): Number of top degrees to retrieve.

    Returns:
    tuple: (top_degrees, top_vertices)
        top_degrees (list): Top k degrees in descending order.
        top_vertices (list): Vertices corresponding to the top k degrees.
    """
    # Calculate degrees by summing along rows
    degrees = get_degrees(adj_matrix)

    # Get indices sorted by degree in descending order
    sorted_indices = np.argsort(degrees)[::-1]

    # Select top k degrees and vertices
    top_vertices = sorted_indices[:k]
    top_degrees = degrees[top_vertices]

    return top_degrees.tolist(), top_vertices.tolist()


def get_degrees(adj_matrix):
    return np.sum(adj_matrix, axis=1)

def is_clique(adj_matrix):
    # all nodes have a degree of n - 1
    if len(adj_matrix) == 1:
        return True
    return np.sum(adj_matrix != 0) == len(adj_matrix) * (len(adj_matrix) - 1)


def greedy_clique(adj_matrix):
    nodes = np.arange(len(adj_matrix))
    mask = np.zeros(len(adj_matrix), dtype=bool)
    # set S as an arbitrary node
    S = np.random.choice(nodes)
    mask[S] = 1
    S = set([S])
    nodes_that_are_not_in_S = set(nodes) - S
    for v in nodes_that_are_not_in_S:
        temp_mask = mask.copy()
        temp_mask[v] = 1
        if is_clique(adj_matrix[temp_mask, :][:, temp_mask]):
            S.add(v)
            mask = temp_mask
    return adj_matrix[mask, :][:, mask]




def least_probable_removal(
        adj,
        compute_prob,
        compute_once=False,
        return_history=False,
        fix_by_spearman=False,
        should_remove_isolated_nodes=False,
):
    history = {
        "removed_nodes": [],
        "probs": [],
        "iterations": [],
        "graph_sizes": [],
        "spearman": [],
        "degrees": [],
    }
    isntance_type_clique = True
    if type(adj) == tuple:
        isntance_type_clique = False

    k = None
    if isntance_type_clique is False:
        # unpack
        should_remove_isolated_nodes = False
        k = adj[0]
        true_vec = adj[2]
        theta = adj[3]
        adj = adj[1]

    R = []
    ratios = []
    ratios_classic = []
    ratio_by_adj = []

    if adj.sum() == 0:
        return np.arange([])

    n = len(adj)
    n_range = np.arange(n)
    mask = np.ones(n).astype(np.True_)
    subgraph = adj
    subgraph_probs_first = compute_prob(subgraph)
    history["probs"].append(subgraph_probs_first)

    condition = lambda subgraph, _: not is_clique(subgraph)
    if isntance_type_clique is False:
        condition = lambda subgraph, mask: (k is None and not is_clique(subgraph)) or (k is not None and mask.sum() > k)
    # while condition(subgraph, mask):
    while (k is None and not is_clique(subgraph)) or (k is not None and mask.sum() > k):
        subgraph = adj[mask, :][:, mask]
        first_iteration = len(R) == 0

        if should_remove_isolated_nodes and any(subgraph.sum(axis=0) == 0):
            # remove isolated nodes from subgraph and mask
            nodes_to_remove = np.where(subgraph.sum(axis=0) == 0)[0]
            nodes_to_remove_in_original = n_range[mask][nodes_to_remove]
            mask[nodes_to_remove_in_original] = False
            subgraph = adj[mask, :][:, mask]
            R.extend(nodes_to_remove_in_original)
            continue
            # add isolated nodes to R
        if compute_once or first_iteration:
            # remove v from subgraph_probs
            subgraph_probs = subgraph_probs_first[mask]
        else:
            subgraph_probs = compute_prob(subgraph)
            degrees = get_degrees(subgraph)

            spearman = 1
            if fix_by_spearman:
                spearman = stats.spearmanr(subgraph_probs, degrees)[0]
                history["spearman"].append(float(np.abs(spearman)))
            history["probs"].append(subgraph_probs)
            # history["degrees"].append(degrees)
            # history["iterations"].append(len(R))
            history["graph_sizes"].append(len(subgraph))

            if fix_by_spearman and spearman < 0:
                subgraph_probs = -1 * subgraph_probs
                # print(stats.spearmanr(subgraph_probs, degrees))

        v = n_range[mask][subgraph_probs.argmin()]
        R.append(v)

        if isntance_type_clique is False:
            subgraph_sum = subgraph.sum(axis=0)
            argsort_subgraph = np.argsort(subgraph_sum)
            original_indices = np.where(mask)[0]
            least = original_indices[argsort_subgraph][0]
            ratios_classic.append(((true_vec[mask] != 0).sum()-(true_vec[least]!=0)) / k)

            subgraph_sum = adj.sum(axis=0)[mask]
            argsort_subgraph = np.argsort(subgraph_sum)
            original_indices = np.where(mask)[0]
            least = original_indices[argsort_subgraph][0]
            ratio_by_adj.append(((true_vec[mask] != 0).sum()-(true_vec[least]!=0)) / k)

        mask[v] = False
        if isntance_type_clique is False:
            ratios.append((true_vec[mask] != 0).sum() / k)

        subgraph = adj[mask, :][:, mask]

    # reverse R
    R.reverse()
    iterations = len(R)
    for v_i, v in enumerate(R):
        mask[v] = True
        if is_clique(adj[mask, :][:, mask]):
            pass
        else:
            mask[v] = False
    if return_history:
        R.reverse()
        history["removed_nodes"] = R
        return n_range[mask], history

    # _,_,true_loss, est_loss = compute_reconstruction_loss(adj, true_vec, n_range[mask], theta)
    # _,_,_, no_theta_est_loss = compute_reconstruction_loss(adj, true_vec, n_range[mask], None)
    # random_mask = np.zeros(n, dtype=bool)
    # random_indices = np.random.choice(n, k, replace=False)
    # random_mask[random_indices] = True
    # _,_,_, random_est_loss = compute_reconstruction_loss(adj, true_vec, n_range[random_mask], theta)
    # _,_,_, no_theta_random_est_loss = compute_reconstruction_loss(adj, true_vec, n_range[random_mask], None)
    # assert k >= mask.sum() > 0, "Mask should not be empty and should not exceed k"
    if isntance_type_clique is False:
        return ratios, ratios_classic, ratio_by_adj
    return n_range[mask]




def least_degree_removal(adj, compute_once=False, return_history=False):
    return least_probable_removal(adj, get_degrees, compute_once, return_history=return_history)


def get_eigen(G, k):
    # A = nx.adjacency_matrix(G).toarray()

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(G)

    # Sort eigenvalues and corresponding eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select the kth eigenvalue and its corresponding eigenvector
    eigenvalue_k = eigenvalues[k]
    eigenvector_k = eigenvectors[:, k]
    return eigenvalue_k, eigenvector_k



def aks(A, k):
    n = len(A)
    # find second eiggenvector
    second_eigenvalue, second_eigenvector = get_eigen(A, 1)
    # take absolute value of each element in the eigenvector
    x = [abs(i) for i in second_eigenvector.tolist()]
    # get the indices of the k largest elements in x
    largest_k = sorted(x, reverse=True)[:k]
    W = [x.index(l) for l in largest_k]
    adj = np.array([1 if i in W else 0 for i in range(n)])
    A_adj = (A @ adj).tolist()
    return [i for i in range(n) if A_adj[i] >= 0.75 * k]


