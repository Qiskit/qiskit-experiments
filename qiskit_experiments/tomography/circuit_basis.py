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
Pauli basis tomography preparation and measurement circuits
"""
from abc import ABC, abstractmethod
from typing import Iterable, Callable, Optional, List
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import HGate, SGate, SdgGate, XGate
from qiskit.exceptions import QiskitError


class CircuitBasis(ABC):
    """A circuit basis generator"""

    def __init__(self, num_elements: int, name: Optional[str] = None):
        """Initialize a basis generator.

        Args:
            num_elements: the number of elements in the basis.
            name: Optional, name for the basis. If None the class
                  name will be used.
        """
        self._num_elements = num_elements
        self._name = name if name else type(self).__name__

    @property
    def name(self) -> str:
        """Return the basis name"""
        return self._name

    def __len__(self) -> int:
        return self._num_elements

    def __call__(self, element: Iterable[int]) -> QuantumCircuit:
        """Return a basis rotation circuit"""
        circuit = QuantumCircuit(len(element), name=f"{self._name}_{element}")
        for qubit, elt in enumerate(element):
            if elt >= self._num_elements:
                raise QiskitError("Invalid basis element index")
            self._append_element(circuit, elt, [qubit])
        return circuit

    @abstractmethod
    def _append_element(self, circuit: QuantumCircuit, i: int, qubits: List[int]):
        """Append a basis element to circuit on the specified qubits"""


class CustomCircuitBasis(CircuitBasis):
    """A Pauli basis circuit generator"""

    def __init__(self, num_elements: int, func: Callable, name: Optional[str] = None):
        """Initialize a basis generator.

        Args:
            num_elements: the number of elements in the basis.
            func: The generator function. It should have signature
                  ``func(element: int) -> QuantumCircuit``
            name: Optional, name for the basis. If None the class
                  name will be used.
        """
        self._func = func
        super().__init__(num_elements, name=name)

    def _append_element(self, circuit: QuantumCircuit, i: int, qubits: List[int]):
        """Return a basis element"""
        circuit.append(self._func(i), qubits)


class PauliMeasCircuitBasis(CircuitBasis):
    """A Pauli measurement basis"""

    def __init__(self):
        super().__init__(3, name="PauliMeas")

    def _append_element(self, circuit: QuantumCircuit, i: int, qubits: List[int]):
        if i == 0:  # Z
            pass
        elif i == 1:  # X
            circuit.append(HGate(), qubits)
        elif i == 2:  # Y
            circuit.append(SdgGate(), qubits)
            circuit.append(HGate(), qubits)


class Pauli4PrepCircuitBasis(CircuitBasis):
    """A 4-element Pauli preparation basis"""

    def __init__(self):
        super().__init__(4, name="Pauli4Prep")

    def _append_element(self, circuit: QuantumCircuit, i: int, qubits: List[int]):
        if i == 0:  # |0>
            pass
        elif i == 1:  # |1>
            circuit.append(XGate(), qubits)
        elif i == 2:  # |+>
            circuit.append(HGate(), qubits)
        elif i == 3:  # |+i>
            circuit.append(HGate(), qubits)
            circuit.append(SGate(), qubits)


class Pauli6PrepCircuitBasis(CircuitBasis):
    """A 6-element Pauli preparation basis"""

    def __init__(self):
        super().__init__(6, name="Pauli6Prep")

    def _append_element(self, circuit: QuantumCircuit, i: int, qubits: List[int]):
        if i == 0:  # |0>
            pass
        elif i == 1:  # |1>
            circuit.append(XGate(), qubits)
        elif i == 2:  # |+>
            circuit.append(HGate(), qubits)
        elif i == 3:  # |->
            circuit.append(XGate(), qubits)
            circuit.append(HGate(), qubits)
        elif i == 4:  # |+i>
            circuit.append(HGate(), qubits)
            circuit.append(SGate(), qubits)
        elif i == 5:  # |+i>
            circuit.append(HGate(), qubits)
            circuit.append(SdgGate(), qubits)
