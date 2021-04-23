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
Quantum process tomography experiment
"""

from typing import Union, Optional, Iterable, List, Tuple
from itertools import product
from qiskit.circuit import QuantumCircuit

from qiskit_experiments.base_experiment import BaseExperiment
from .basis import PauliMeasurementBasis, PauliPreparationBasis
from .qpt_analysis import QPTAnalysis


class QPTExperiment(BaseExperiment):
    """Quantum process tomography experiment"""

    __analysis_class__ = QPTAnalysis

    def __init__(
        self,
        circuit: Union[QuantumCircuit, "InstructionLike"],
        prep_qubits: Optional[Iterable[int]] = None,
        meas_qubits: Optional[Iterable[int]] = None,
        qubits: Optional[Iterable[int]] = None,
        basis_elements: Optional[Iterable[Tuple[List[int], List[int]]]] = None,
        meas_basis: "TomographyBasis" = PauliMeasurementBasis(),
        prep_basis: "TomographyBasis" = PauliPreparationBasis(),
    ):
        """Initialize a state tomography experiment.

        Args:
            circuit: the quantum process circuit. If not a quantum circuit
                it must be a class that can be appended to a quantum circuit.
            qubits: Optional, the physical qubits for the initial state circuit.
            meas_basis: Tomography basis for measurements. If not specified the
                        default basis is the :class:`PauliMeasurementBasis`.
            meas_qubits: Optional, the qubits to be measured. These should refer
                to the logical qubits in the state circuit. If None all qubits
                in the state circuit will be measured.
            prep_basis: Tomography basis for measurements. If not specified the
                        default basis is the :class:`PauliPreparationBasis`.
            prep_qubits: Optional, the qubits to be prepared. These should refer
                to the logical qubits in the process circuit. If None all qubits
                in the process circuit will be prepared.
            basis_elements: Optional, the basis elements to be measured. If None
                All basis elements will be measured. If specified each element
                is given by a a pair of lists
                ``([m[0], m[1], ...], [p[0], p[1], ..])`` where ``m[i]`` and
                ``p[i]`` are the measurement basis and preparation basis indices
                respectively for qubit-i.
        """
        num_qubits = circuit.num_qubits
        # Get physical qubits
        if qubits is None:
            qubits = num_qubits

        # Get prepared qubits
        if prep_qubits is None:
            self._prep_qubits = tuple(range(num_qubits))
        else:
            self._prep_qubits = tuple(prep_qubits)

        # Get measured qubits
        if meas_qubits is None:
            self._meas_qubits = tuple(range(num_qubits))
        else:
            self._meas_qubits = tuple(meas_qubits)

        # Bases
        self._prep_basis = prep_basis
        self._meas_basis = meas_basis

        # Get the target tomography circuit
        if isinstance(circuit, QuantumCircuit):
            target_circuit = circuit
        else:
            # Convert input to a circuit
            target_circuit = QuantumCircuit(num_qubits)
            target_circuit.append(circuit, range(num_qubits))
        self._circuit = target_circuit

        # Store custom basis elements
        self._custom_basis_elements = basis_elements
        super().__init__(qubits, circuit_options=set(["basis_elements"]))

    # pylint: disable = arguments-differ
    def circuits(self, backend=None, basis_elements=None):

        # Get basis elements for measurement circuits
        if basis_elements is None:
            basis_elements = self._basis_elements()

        # Get qubits and clbits
        total_clbits = self._circuit.num_clbits + len(self._meas_qubits)
        circ_qubits = list(range(self._circuit.num_qubits))
        circ_clbits = list(range(self._circuit.num_clbits))
        meas_clbits = list(range(self._circuit.num_clbits, total_clbits))
        num_outcomes = self._meas_basis.circuit.num_outcomes

        # Build circuits
        circuits = []
        for meas_element, prep_element in basis_elements:
            circ = QuantumCircuit(
                self.num_qubits, total_clbits, name=f"{self._type}_{meas_element}_{prep_element}"
            )

            # Add tomography preparation
            circ.reset(self._prep_qubits)
            circ.append(self._prep_basis.circuit(prep_element), self._prep_qubits)
            circ.barrier(self._prep_qubits)

            # Add target circuit
            circ.append(self._circuit, circ_qubits, circ_clbits)

            # Add tomography measurement
            circ.barrier(self._meas_qubits)
            circ.append(self._meas_basis.circuit(meas_element), self._meas_qubits)
            circ.measure(self._meas_qubits, meas_clbits)

            # Shifted element for including different measurement outcomes
            shifted_meas_element = [num_outcomes * i for i in meas_element]

            # Add metadata
            circ.metadata = {
                "experiment_type": self._type,
                "qubits": self.physical_qubits,
                "clbits": meas_clbits,
                "m_idx": shifted_meas_element,
                "p_idx": list(prep_element),
                "m_basis": self._meas_basis.name,
                "p_basis": self._prep_basis.name,
            }
            circuits.append(circ)
        return circuits

    def _basis_elements(self):
        """Return basis elements"""
        if self._custom_basis_elements is not None:
            return self._custom_basis_elements
        meas_elements = product(range(len(self._meas_basis.circuit)), repeat=len(self._meas_qubits))
        prep_elements = product(range(len(self._prep_basis.circuit)), repeat=len(self._prep_qubits))
        return product(meas_elements, prep_elements)
