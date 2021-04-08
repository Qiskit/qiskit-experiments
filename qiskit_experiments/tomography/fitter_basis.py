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
from typing import Iterable, Optional
import numpy as np


class FitterBasis:
    """A operator basis for fitting"""

    def __init__(self, elements: np.ndarray, name: Optional[str] = None):
        """Initialize a basis generator.

        Args:
            elements: array of element matrices
            name: Optional, name for the basis. If None the class
                  name will be used.
        """
        self._num_elements = len(elements)
        self._name = name if name else type(self).__name__
        self._elements = elements

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


class PauliMeasBasis(FitterBasis):
    """A Pauli PVM measurement basis"""

    def __init__(self):
        povms = np.array(
            [
                [[1, 0], [0, 0]],
                [[0, 0], [0, 1]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
                [[0.5, -0.5j], [0.5j, 0.5]],
                [[0.5, 0.5j], [-0.5j, 0.5]],
            ]
        )
        super().__init__(povms, name="PauliMeas")


class Pauli4PrepBasis(FitterBasis):
    """A 4-element Pauli preparation basis"""

    def __init__(self):
        povms = np.array(
            [
                [[1, 0], [0, 0]],
                [[0, 0], [0, 1]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, -0.5j], [0.5j, 0.5]],
            ]
        )
        super().__init__(povms, name="Pauli4Prep")


class Pauli6PrepBasis(FitterBasis):
    """A 6-element Pauli preparation basis"""

    def __init__(self):
        povms = np.array(
            [
                [[1, 0], [0, 0]],
                [[0, 0], [0, 1]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, -0.5], [-0.5, 0.5]],
                [[0.5, -0.5j], [0.5j, 0.5]],
                [[0.5, 0.5j], [-0.5j, 0.5]],
            ]
        )
        super().__init__(povms, name="Pauli6Prep")
