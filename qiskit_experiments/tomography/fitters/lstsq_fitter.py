# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Uncontrained linear least-squares tomography fitter.
"""

from typing import Optional
import numpy as np
from scipy.linalg import lstsq, eigh

from qiskit_experiments.base_analysis import AnalysisResult


def lstsq_tomography_fit(
    basis_matrix: np.ndarray,
    data: np.ndarray,
    psd: bool = True,
    trace: Optional[float] = None,
    **kwargs,
) -> AnalysisResult:
    r"""
    Reconstruct a density matrix using MLE least-squares fitting.

    Args:
        basis_matrix: lstsq matrix `a`, Stacked basis matrix of vectorized
                      basis POVMs
        data: lstsq vector `b`, 1D array of basis element expectation value
        psd: Enforced the fitted matrix to be positive
            semidefinite (default: True)
        trace: trace constraint for the fitted matrix (default: None).
        kwargs: additional kwargs for scipy.linalg.lstsq

    Raises:
        ValueError: If the fitted vector is not a square matrix

    Returns:
        The fitted matrix rho that minimizes
        :math:`||\text{basis_matrix} \cdot
         \text{vec}(\text{rho}) - \text{data}||_2`.

    Additional Information:

        Objective function
        ------------------
        This fitter solves the least-squares minimization:

            minimize :math:`||a \cdot x - b ||_2`

        where:
            a is the matrix of measurement operators a[i] = vec(M_i).H
            b is the vector of expectation value data for each projector
              b[i] ~ Tr[M_i.H * x] = (a * x)[i]
            x is the vectorized density matrix (or Choi-matrix) to be fitted

        PSD Constraint
        --------------
        Since this minimization problem is unconstrained the returned fitted
        matrix may not be postive semidefinite (PSD). To enforce the PSD
        constraint the fitted matrix is rescaled using the method proposed in
        Reference [1].

        Trace constraint
        ----------------
        In general the trace of the fitted matrix will be determined by the
        input data. If a trace constraint is specified the fitted matrix
        will be rescaled to have this trace by:
            :math:`\text{rho} = \frac{\text{trace}\cdot\text{rho}}
            {\text{trace}(\text{rho})}`

    References:
        [1] J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502
            (2012). Open access: arXiv:1106.5458 [quant-ph].
    """
    # Perform least squares fit using Scipy.linalg lstsq function
    lstsq_opts = {"check_finite": False, "lapack_driver": "gelsy"}
    for key, val in kwargs.items():
        lstsq_opts[key] = val
    sol, residues, rank, svals = lstsq(basis_matrix, data, **lstsq_opts)

    # Reshape fit to a density matrix
    size = len(sol)
    dim = int(np.sqrt(size))
    if dim * dim != size:
        raise ValueError("fitted vector is not a square matrix.")
    rho_fit = np.reshape(sol, (dim, dim), order="F")

    # Rescale fitted density matrix be positive-semidefinite
    if psd is True:
        rho_fit = make_positive_semidefinite(rho_fit)

    if trace is not None:
        rho_fit *= trace / np.trace(rho_fit)

    analysis_result = AnalysisResult(
        {"value": rho_fit, "fit": {"residues": residues, "rank": rank, "singular_values": svals}}
    )
    return analysis_result


def make_positive_semidefinite(mat: np.array, epsilon: Optional[float] = 0) -> np.array:
    """
    Rescale a Hermitian matrix to nearest postive semidefinite matrix.

    Args:
        mat: a hermitian matrix.
        epsilon: (default: 0) the threshold for setting
            eigenvalues to zero. If epsilon > 0 positive eigenvalues
            below epsilon will also be set to zero.
    Raises:
        ValueError: If epsilon is negative
    Returns:
        The input matrix rescaled to have non-negative eigenvalues.

    References:
        [1] J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502
            (2012). Open access: arXiv:1106.5458 [quant-ph].
    """

    if epsilon < 0:
        raise ValueError("epsilon must be non-negative.")

    # Get the eigenvalues and eigenvectors of rho
    # eigenvalues are sorted in increasing order
    # v[i] <= v[i+1]

    dim = len(mat)
    v, w = eigh(mat)
    for j in range(dim):
        if v[j] < epsilon:
            tmp = v[j]
            v[j] = 0.0
            # Rescale remaining eigenvalues
            x = 0.0
            for k in range(j + 1, dim):
                x += tmp / (dim - (j + 1))
                v[k] = v[k] + tmp / (dim - (j + 1))

    # Build positive matrix from the rescaled eigenvalues
    # and the original eigenvectors

    mat_psd = np.zeros([dim, dim], dtype=complex)
    for j in range(dim):
        mat_psd += v[j] * np.outer(w[:, j], np.conj(w[:, j]))

    return mat_psd
