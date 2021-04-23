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
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import HGate, XGate, ZGate, SGate, SdgGate
from .basis import TomographyBasis, TomographyCircuitBasis, TomographyMatrixBasis


class PauliMeasurementCircuitBasis(TomographyCircuitBasis):
    """A Pauli measurement basis"""

    def __init__(self):
        # Z-meas rotation
        meas_z = QuantumCircuit(1, name="PauliMeasZ")
        # X-meas rotation
        meas_x = QuantumCircuit(1, name="PauliMeasX")
        meas_x.append(HGate(), [0])
        # Y-meas rotation
        meas_y = QuantumCircuit(1, name="PauliMeasY")
        meas_y.append(SdgGate(), [0])
        meas_y.append(HGate(), [0])
        super().__init__([meas_z, meas_x, meas_y], num_outcomes=2, name="PauliMeas")


class PauliMeasurementMatrixBasis(TomographyMatrixBasis):
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
        super().__init__(povms, name=type(self).__name__)


class PauliMeasurementBasis(TomographyBasis):
    """Pauli measurement tomography basis"""

    def __init__(self):
        """Pauli measurement tomography basis"""
        super().__init__(
            PauliMeasurementCircuitBasis(), PauliMeasurementMatrixBasis(), name=type(self).__name__
        )


class PauliPreparationCircuitBasis(TomographyCircuitBasis):
    """A Pauli measurement basis"""

    def __init__(self):
        # |0> Zp rotation
        prep_zp = QuantumCircuit(1, name="PauliPrepZp")
        # |1> Zm rotation
        prep_zm = QuantumCircuit(1, name="PauliPrepZp")
        prep_zm.append(XGate(), [0])
        # |+> Xp rotation
        prep_xp = QuantumCircuit(1, name="PauliPrepXp")
        prep_xp.append(HGate(), [0])
        # |+i> Yp rotation
        prep_yp = QuantumCircuit(1, name="PauliPrepYp")
        prep_yp.append(HGate(), [0])
        prep_yp.append(SGate(), [0])
        super().__init__([prep_zp, prep_zm, prep_xp, prep_yp], name="PauliPrep")


class PauliPreparationMatrixBasis(TomographyMatrixBasis):
    """Minimum 4-element Pauli preparation basis"""

    def __init__(self):
        povms = np.array(
            [
                [[1, 0], [0, 0]],
                [[0, 0], [0, 1]],
                [[0.5, 0.5], [0.5, 0.5]],
                [[0.5, -0.5j], [0.5j, 0.5]],
            ]
        )
        super().__init__(povms, name="PauliPrep")


class PauliPreparationBasis(TomographyBasis):
    """Minimum 4-element Pauli preparation basis"""

    def __init__(self):
        """Pauli measurement tomography basis"""
        super().__init__(
            PauliPreparationCircuitBasis(), PauliPreparationMatrixBasis(), type(self).__name__
        )


class Pauli6PreparationCircuitBasis(TomographyCircuitBasis):
    """A 6-element Pauli preparation basis"""

    def __init__(self):
        # |0> Zp rotation
        prep_zp = QuantumCircuit(1, name="PauliPrepZp")
        # |1> Zm rotation
        prep_zm = QuantumCircuit(1, name="PauliPrepZp")
        prep_zm.append(XGate(), [0])
        # |+> Xp rotation
        prep_xp = QuantumCircuit(1, name="PauliPrepXp")
        prep_xp.append(HGate(), [0])
        # |+i> Yp rotation
        prep_yp = QuantumCircuit(1, name="PauliPrepYp")
        prep_yp.append(HGate(), [0])
        prep_yp.append(SGate(), [0])
        # |-> Xm rotation
        prep_xm = QuantumCircuit(1, name="PauliPrepXp")
        prep_xm.append(HGate(), [0])
        prep_xm.append(ZGate(), [0])
        # |-i> Ym rotation
        prep_ym = QuantumCircuit(1, name="PauliPrepYp")
        prep_ym.append(HGate(), [0])
        prep_ym.append(SdgGate(), [0])
        super().__init__([prep_zp, prep_zm, prep_xp, prep_yp])


class Pauli6PreparationMatrixBasis(TomographyMatrixBasis):
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
        super().__init__(povms, name=type(self).__name__)


class Pauli6PreparationBasis(TomographyBasis):
    """Pauli-6 preparation tomography basis"""

    def __init__(self):
        """Pauli-6 preparation tomography basis"""
        super().__init__(
            Pauli6PreparationCircuitBasis(), Pauli6PreparationMatrixBasis(), "Pauli6Prep"
        )
