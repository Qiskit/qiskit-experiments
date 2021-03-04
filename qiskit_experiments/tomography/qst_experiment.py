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
Quantum state tomography experiment
"""

from typing import Union, Optional, Iterable, List
from itertools import product
from qiskit.circuit import QuantumCircuit

from qiskit_experiments.base_experiment import BaseExperiment
from .circuit_basis import CircuitBasis, PauliMeasCircuitBasis
from .qst_analysis import QSTAnalysis


class QSTExperiment(BaseExperiment):
    """Quantum state tomography experiment"""

    __analysis_class__ = QSTAnalysis

    def __init__(
        self,
        circuit: Union[QuantumCircuit, "InstructionLike"],
        meas_qubits: Optional[Iterable[int]] = None,
        qubits: Optional[Iterable[int]] = None,
        basis_elements: Optional[Iterable[List[int]]] = None,
        meas_basis: Optional[CircuitBasis] = PauliMeasCircuitBasis(),
    ):
        """Initialize a state tomography experiment.

        Args:
            circuit: the quantum state circuit. If not a quantum circuit
                it must be a class that can be appended to a quantum circuit.
            meas_qubits: Optional, the qubits to be measured. These should refer
                to the logical qubits in the state circuit. If None all qubits
                in the state circuit will be measured.
            qubits: Optional, the physical qubits for the initial state circuit.
            basis_elements: Optional, the basis elements to be measured. If None
                All basis elements will be measured.
            meas_basis: Optional, measurement basis circuit generator. It should
                have signature fn(element) -> QuantumCircuit. If None Pauli
                basis measurement is used.
        """
        num_qubits = circuit.num_qubits
        # Get physical qubits
        if qubits is None:
            qubits = num_qubits

        # Get measured qubits
        if meas_qubits is None:
            self._meas_qubits = tuple(range(num_qubits))
        else:
            self._meas_qubits = tuple(meas_qubits)

        # Measurement basis
        self._meas_basis = meas_basis

        # Get initial state preparation circuit
        if isinstance(circuit, QuantumCircuit):
            state_circuit = circuit
        else:
            # Convert input to a circuit
            state_circuit = QuantumCircuit(num_qubits)
            state_circuit.append(circuit, range(num_qubits))
        self._circuit = state_circuit

        # Store custom basis elements
        self._basis_elements = basis_elements
        super().__init__(qubits, circuit_options=set(["basis_elements"]))

    # pylint: disable = arguments-differ
    def circuits(self, backend=None, basis_elements=None):
        circ_qubits = list(range(self._circuit.num_qubits))
        meas_qubits = self._meas_qubits

        num_clbits = self._circuit.num_clbits + len(meas_qubits)
        circ_clbits = list(range(self._circuit.num_clbits))
        meas_clbits = list(range(self._circuit.num_clbits, num_clbits))

        if basis_elements is None:
            if self._basis_elements is None:
                basis_elements = product(
                    range(len(self._meas_basis)), repeat=len(self._meas_qubits)
                )
            else:
                basis_elements = self._basis_elements

        circuits = []
        for element in basis_elements:
            circ = QuantumCircuit(self.num_qubits, num_clbits)
            circ.append(self._circuit, circ_qubits, circ_clbits)
            circ.barrier(range(self.num_qubits))
            circ.append(self._meas_basis(element), meas_qubits)
            circ.measure(meas_qubits, meas_clbits)

            # Shifted element for including different measurement
            # outcomes for each basis (assuming 2 outcome measurements)
            shifted_element = [2 * i for i in element]

            # Add metadata
            circ.metadata = {
                "experiment_type": self._type,
                "qubits": self.physical_qubits,
                "clbits": meas_clbits,
                "m_idx": shifted_element,
                "m_basis": self._meas_basis.name,
            }
            circuits.append(circ)
        return circuits
