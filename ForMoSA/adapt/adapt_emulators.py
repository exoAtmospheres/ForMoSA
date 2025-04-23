from __future__ import print_function, division
import numpy as np
import torch
from torchnmf.nmf import NMF
from sklearn.decomposition import PCA

# ----------------------------------------------------------------------------------------------------------------------


def get_ws(eigenspectra, flx_grid):
    """
    Since we will overflow memory if we actually calculate Phi, we have to
    determine ws in a memory-efficient manner.

    Args:
        - eigenspectra        (np.ndarray): Eigenspectra of the PCA decomposion
        - flx_grid            (np.ndarray): Flux of the normalized grid
    Returns:
        ws                    (np.ndarray): Array of weights

    Author: Miles Lucas
    """
    m = len(eigenspectra)
    M = len(flx_grid)
    ws = np.empty((M * m,))
    for i in range(m):
        for j in range(M):
            ws[i * M + j] = eigenspectra[i].T @ flx_grid[j]

    return ws


# ----------------------------------------------------------------------------------------------------------------------


def emulator_PCA(ds, PCA_comp = 'NA'):
    """
    Emulator of a grid spectra with PCA decomposition
    
    Args:
        ds                      (xarray): Grid used for the PCA
        PCA_comp          (str or float): Number of PCA component to use. 
                                          'NA' by default means that the emulator will try to explain at least 99% of the variance of the grid
    Returns:
        flx_grid_mean       (np.ndarray): Mean flux of the normalized grid
        flx_grid_std        (np.ndarray): Standart deviation of the normalized grid
        PCA_vectors         (np.ndarray): Eigenspectra of the PCA decomposion
        PCA_weights             (xarray): Grid of normalization factors and weights (format = (PCA_comp, grid_shape))

    Author: Matthieu Ravet, adapted from https://iopscience.iop.org/article/10.1088/0004-637X/812/2/128/pdf  
    """

    # Extract grid
    grid = ds['grid']
    attr = ds.attrs
    attr.pop("res")
    ds.close()
    flx_grid = grid.to_numpy()
    og_grid = np.copy(flx_grid)

    # Reshape the grid to be 2D and in the right dimension
    flx_grid = flx_grid.reshape(flx_grid.shape[0], -1).T

    # Normalize to an average of 1 to remove uninteresting correlation
    nfs = flx_grid.mean(1)
    flx_grid /= nfs[:, np.newaxis]
    # Center and whiten
    flx_grid_mean = flx_grid.mean(0)
    flx_grid -= flx_grid_mean
    flx_grid_std = flx_grid.std(0)
    flx_grid /= flx_grid_std

    # Perform PCA using sklearn
    if PCA_comp == 'NA': # Will automatically choose the number of component explaining at leat 99% of the variance of the grid
        default_pca_kwargs = dict(n_components=0.99, svd_solver="full")
    else:
        default_pca_kwargs = dict(n_components=PCA_comp, svd_solver="full")
    pca = PCA(**default_pca_kwargs)
    pca.fit_transform(flx_grid)
    vectors = pca.components_

    # Extract wheights
    ws = get_ws(vectors, flx_grid)

    # - - - - - - - -

    # Reshape weights and norm_factor and merge them
    ws = ws.reshape((pca.n_components_,) + og_grid.shape[1:])
    nfs = nfs.reshape(og_grid.shape[1:])
    weights = np.concatenate((nfs[np.newaxis,:], ws), axis=0)

    return flx_grid_mean, flx_grid_std, vectors, weights


# ----------------------------------------------------------------------------------------------------------------------


def emulator_NMF(ds, NMF_comp):
    """
    Emulator of a grid spectra with NMF decomposition. Need CUDA to work on GPU
    
    Args:
        ds                      (xarray): Grid used for the NMF
        PCA_comp          (str or float): Number of NMF component to use.
    Returns:
        NMF_spectra         (np.ndarray): Reformated H matrix of the NMF decomposion = List of eigenspectra
        weights                 (xarray): Reformated W matrix of the NMF decomposion = Grid of weights (format = (NMF_comp, grid_shape))

    Author: Matthieu Ravet
    """
    # DEVICE SETUP 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)  # For reproducibility of the NMF decomposition

    # Extract grid
    grid = ds['grid']
    attr = ds.attrs
    attr.pop("res")
    ds.close()
    flx_grid = grid.to_numpy()
    og_grid = np.copy(flx_grid)

    # Reshape the grid to be 2D and in the right dimension
    flx_grid = flx_grid.reshape(flx_grid.shape[0], -1)

    # Model with NNF (Non-Negative matrix Factorization)
    flx_grid_torch = torch.tensor(flx_grid, device=device, requires_grad=False, dtype=torch.float64)

    # Model
    model = NMF(flx_grid_torch.shape, rank=NMF_comp).to(device)
    model.fit(flx_grid_torch, max_iter=10000)

    # Extract weights and eigenspectra
    H = model.H.cpu().detach().numpy()
    W = model.W.cpu().detach().numpy()

    # Convert/reformat them
    vectors = H.T
    weights = W.T.reshape((NMF_comp,) + og_grid.shape[1:])

    return vectors, weights