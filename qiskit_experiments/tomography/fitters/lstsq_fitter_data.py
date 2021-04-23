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
Data processing for linear least square tomography fitters
"""

import numpy as np


def qst_basis_matrix(
    meas_basis_data: np.ndarray, meas_matrix_basis: "TomographyMatrixBasis"
) -> np.ndarray:
    """Return stacked vectorized basis matrix A for least squares."""
    size, msize1 = meas_basis_data.shape
    mdim = meas_matrix_basis.dim ** (2 * msize1)
    ret = np.zeros((size, mdim), dtype=complex)

    for i in range(size):
        m_op = meas_matrix_basis(meas_basis_data[i])
        ret[i] = np.ravel(m_op, order="F")
    return ret


def qpt_basis_matrix(
    meas_basis_data: np.ndarray,
    prep_basis_data: np.ndarray,
    meas_matrix_basis: "TomographyMatrixBasis",
    prep_matrix_basis: "TomographyMatrixBasis",
) -> np.ndarray:
    """Return stacked vectorized basis matrix A for least squares."""
    size, msize1 = meas_basis_data.shape
    _, psize1 = prep_basis_data.shape
    mdim = meas_matrix_basis.dim ** (2 * msize1)
    pdim = prep_matrix_basis.dim ** (2 * psize1)
    ret = np.zeros((size, mdim * pdim), dtype=complex)

    for i in range(size):
        m_op = meas_matrix_basis(meas_basis_data[i])
        p_op = prep_matrix_basis(prep_basis_data[i])
        ret[i] = np.ravel(np.kron(p_op.T, m_op), order="F")
    return ret


def hedged_binomial_weights(
    frequencies: np.ndarray, shots: np.ndarray, num_qubits: int, beta: float = 0.5
) -> np.array:
    """
    Compute binomial weights for list or dictionary of counts.

    Args:
        frequencies: Array of count frequences for each basis element.
        shots: Array of total shots for each basis element for
               converting frequencies to probabilities
        num_qubits: The number of qubits measured.
        beta: (default: 0.5) A nonnegative hedging parameter used to bias
            probabilities computed from input counts away from 0 or 1.

    Returns:
        np.ndarray: The binomial weights for the input counts and beta parameter.

    Raises:
        ValueError: In case beta is negative.

    Additional Information:

        The weights are determined by
            w[i] = sqrt(shots / p[i] * (1 - p[i]))
            p[i] = (counts[i] + beta) / (shots + K * beta)
        where
            `shots` is the sum of all counts in the input
            `p` is the hedged probability computed for a count
            `K` is the total number of possible measurement outcomes.
    """
    if beta < 0:
        raise ValueError("beta = {} must be non-negative.".format(beta))

    outcomes_num = 2 ** num_qubits
    # Compute hedged frequencies which are shifted to never be 0 or 1.
    probs_hedged = (frequencies + beta) / (shots + outcomes_num * beta)

    # Return gaussian weights for 2-outcome measurements.
    return np.sqrt(shots / (probs_hedged * (1 - probs_hedged)))
