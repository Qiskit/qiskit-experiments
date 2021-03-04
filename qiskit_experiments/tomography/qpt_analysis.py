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
Quantum process tomography analysis
"""
from typing import List, Dict, Tuple
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.result import marginal_counts
from qiskit.quantum_info import Choi

from qiskit_experiments.base_analysis import BaseAnalysis
from qiskit_experiments.tomography.fitter_basis import PauliMeasBasis, Pauli4PrepBasis
from qiskit_experiments.tomography.fitters import (
    qpt_basis_matrix,
    hedged_binomial_weights,
    lstsq_tomography_fit,
    cvx_lstsq_tomography_fit,
    CVXSolverChecker,
)


class QPTAnalysis(BaseAnalysis):
    """Quantum process tomography experiment analysis."""

    # pylint: disable = arguments-differ
    def _run_analysis(
        self,
        experiment_data,
        method: str = "auto",
        psd: bool = True,
        trace: float = None,
        trace_preserving: bool = True,
        binomial_weights: bool = True,
        beta: float = 0.5,
        meas_basis: "FitterBasis" = None,
        prep_basis: "FitterBasis" = None,
        **kwargs,
    ):
        # Tomography fitter basis
        # TODO: get built in bases from metadata
        if meas_basis is None:
            meas_basis = PauliMeasBasis()
        if prep_basis is None:
            prep_basis = Pauli4PrepBasis()

        # Choose automatic method
        if method == "auto":
            if CVXSolverChecker().has_sdp_solver_not_scs:
                # We don't use the SCS solver for automatic method as it has
                # lower accuracy than the other supported SDP solvers which
                # typically results in the returned matrix not being
                # completely positive.
                method = "cvx"
            else:
                method = "lstsq"

        # Extract tomography measurement data
        # Extract tomography measurement data
        mbasis_data, pbasis_data, freq_data, shot_data = self._measurement_data(
            experiment_data.data
        )
        basis_matrix = qpt_basis_matrix(mbasis_data, pbasis_data, meas_basis, prep_basis)
        prob_data = freq_data / shot_data
        num_qubits = len(mbasis_data[0])

        # Optionally apply a weights vector to the data and projectors
        if binomial_weights:
            weights = hedged_binomial_weights(freq_data, shot_data, num_qubits, beta=beta)
            basis_matrix = weights[:, None] * basis_matrix
            prob_data = weights * prob_data

        # Run fitter
        if method == "lstsq":
            # Least Square does not support trace preserving condition
            if trace_preserving:
                trace = 2 ** num_qubits
            else:
                trace = None
            result = lstsq_tomography_fit(basis_matrix, prob_data, psd=psd, trace=trace, **kwargs)
        elif method == "cvx":
            result = cvx_lstsq_tomography_fit(
                basis_matrix, prob_data, psd=psd, trace=trace, trace_preserving=True, **kwargs
            )
        else:
            raise QiskitError("Unrecognized QSTAnalysis method {}".format(method))

        # Update returned data type to be DensityMatrix
        result["value"] = Choi(result["value"])

        return result, None

    @staticmethod
    def _measurement_data(data: List[Dict[str, any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return list a tuple of basis, frequency, shot data"""
        freq_dict = {}
        shot_dict = {}
        meas_size = None
        prep_size = None

        for datum in data:
            metadata = datum["metadata"]
            counts = marginal_counts(datum["counts"], metadata["clbits"])
            shots = sum(counts.values())
            meas_element = tuple(metadata["m_idx"])
            prep_element = tuple(metadata["p_idx"])

            if meas_size is None:
                meas_size = len(meas_element)
            if prep_size is None:
                prep_size = len(prep_element)

            # Get count data
            for key, freq in counts.items():
                outcome_element = list(meas_element)
                for i, outcome in enumerate(reversed(key)):
                    outcome_element[i] += int(outcome)
                element = tuple(outcome_element), prep_element
                if element in freq_dict:
                    freq_dict[element] += freq
                    shot_dict[element] += shots
                else:
                    freq_dict[element] = freq
                    shot_dict[element] = shots

        num_elements = len(freq_dict)

        meas_basis_data = np.zeros((num_elements, meas_size), dtype=int)
        prep_basis_data = np.zeros((num_elements, prep_size), dtype=int)
        freq_data = np.zeros(num_elements, dtype=int)
        shot_data = np.zeros(num_elements, dtype=int)

        for i, (key, val) in enumerate(freq_dict.items()):
            meas_basis_data[i] = key[0]
            prep_basis_data[i] = key[1]
            freq_data[i] = val
            shot_data[i] = shot_dict[key]

        return meas_basis_data, prep_basis_data, freq_data, shot_data
