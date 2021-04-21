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
Circuit basis for tomography preparation and measurement circuits
"""
from typing import Iterable, Optional, List
import numpy as np
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info import Operator
from qiskit.exceptions import QiskitError


class TomographyBasis:
    """A tomography basis"""

    def __init__(
        self,
        circuit_basis: "TomographyCircuitBasis",
        matrix_basis: "TomographyMatrixBasis",
        name: Optional[str] = None,
    ):
        """Initialize a tomography basis"""
        self._circuit_basis = circuit_basis
        self._matrix_basis = matrix_basis
        self._name = name

    @property
    def name(self) -> str:
        """Return the basis name"""
        return self._name

    @property
    def circuit(self):
        """Return circuit basis"""
        return self._circuit_basis

    @property
    def matrix(self):
        """Return matrix basis"""
        return self._matrix_basis


class TomographyCircuitBasis:
    """A circuit generator tomography experiments bases."""

    def __init__(
        self, instructions: List[Instruction], num_outcomes: int = 0, name: Optional[str] = None
    ):
        """Initialize a circuit generator.

        Args:
            instructions: list of instructions for basis rotations.
            num_outcomes: the number of outcomes for each basis element.
            name: Optional, name for the basis. If None the class
                  name will be used.

        Raises:
            QiskitError: if input instructions are not valid.
        """
        # Convert inputs to quantum circuits
        self._instructions = [self._convert_input(i) for i in instructions]
        self._num_outcomes = num_outcomes
        self._name = name if name else type(self).__name__

        # Check number of qubits
        self._num_qubits = self._instructions[0].num_qubits
        for i in self._instructions[1:]:
            if i.num_qubits != self._num_qubits:
                raise QiskitError(
                    "Invalid input instructions. All instructions must be"
                    " defined on the same number of qubits."
                )

    @property
    def name(self) -> str:
        """Return the basis name"""
        return self._name

    @property
    def num_outcomes(self) -> int:
        """Return the number of outcomes"""
        return self._num_outcomes

    @property
    def num_qubits(self) -> int:
        """Return the number of qubits of the basis"""
        return self._num_qubits

    def __len__(self) -> int:
        return len(self._instructions)

    def __call__(self, element: Iterable[int]) -> QuantumCircuit:
        """Return a composite basis rotation circuit.

        Args:
            element: a list of basis elements to tensor together.

        Returns:
            the rotation circuit for the specified basis

        Raises:
            QiskitError: if the specified elements are invalid.
        """
        num_qubits = len(element) * self.num_qubits
        circuit = QuantumCircuit(num_qubits, name=f"{self._name}_{element}")
        for i, elt in enumerate(element):
            if elt >= len(self):
                raise QiskitError("Invalid basis element index")
            qubits = list(range(i * self.num_qubits, (i + 1) * self.num_qubits))
            circuit.append(self._instructions[elt], qubits)
        return circuit

    @staticmethod
    def _convert_input(unitary):
        """Convert input to an Instruction"""
        if isinstance(unitary, Instruction):
            return unitary
        if hasattr(unitary, "to_instruction"):
            return unitary.to_instruction()
        return Operator(unitary).to_instruction()


class TomographyMatrixBasis:
    """A operator basis for tomography fitters"""

    def __init__(self, elements: np.ndarray, name: Optional[str] = None):
        """Initialize a basis generator.

        Args:
            elements: array of element matrices
            name: Optional, name for the basis. If None the class
                  name will be used.
        """
        self._num_elements = len(elements)
        self._name = name if name else type(self).__name__
        self._elements = np.asarray(elements, dtype=complex)

    @property
    def name(self) -> str:
        """Return the basis name"""
        return self._name

    def __len__(self) -> int:
        return self._num_elements

    @property
    def dim(self) -> int:
        """Return the dimension of a basis element"""
        return self._elements.shape[1]

    def __call__(self, element: Iterable[int]) -> np.ndarray:
        """Return a basis rotation circuit"""
        ret = np.ones(1)
        for i in element:
            ret = np.kron(self._elements[i], ret)
        return ret
