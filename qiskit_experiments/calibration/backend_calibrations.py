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

"""Store and manage the results of calibration experiments in the context of a backend."""

from datetime import datetime
from typing import List
import copy

from qiskit.providers.backend import BackendV1 as Backend
from qiskit.circuit import Parameter
from qiskit_experiments.calibration.calibrations_manager import CalibrationsManager, ParameterKey


class BackendCalibrationsManager(CalibrationsManager):
    """
    A Calibrations class to enable a seamless interplay with backend objects.
    This class enables users to export their calibrations into a backend object.
    Additionally, it creates frequency parameters for qubits and readout resonators.
    The parameters are named `qubit_lo_freq` and `meas_lo_freq` to be consistent
    with the naming in backend.defaults(). These two parameters are not attached to
    any schedule.
    """

    def __init__(self, backend: Backend):
        """Setup an instance to manage the calibrations of a backend."""
        super().__init__(backend.configuration().control_channels)

        # Use the same naming convention as in backend.defaults()
        self.qubit_freq = Parameter("qubit_lo_freq")
        self.meas_freq = Parameter("meas_lo_freq")
        self._register_parameter(self.qubit_freq)
        self._register_parameter(self.meas_freq)

        self._qubits = set(range(backend.configuration().n_qubits))
        self._backend = backend

    def get_frequencies(
        self,
        meas_freq: bool,
        group: str = "default",
        cutoff_date: datetime = None,
    ) -> List[float]:
        """
        Get the most recent qubit or measurement frequencies. These frequencies can be
        passed to the run-time options of :class:`BaseExperiment`. If no calibrated value
        for the frequency of a qubit is found then the default value from the backend
        defaults is used. Only valid parameter values are returned.

        Args:
            meas_freq: If True return the measurement frequencies otherwise return the qubit
                frequencies.
            group: The calibration group from which to draw the
                parameters. If not specified, this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values
                that may be erroneous.

        Returns:
            A List of qubit or measurement frequencies for all qubits of the backend.
        """

        param = self.meas_freq.name if meas_freq else self.qubit_freq.name

        freqs = []
        for qubit in self._qubits:
            if ParameterKey(None, param, (qubit,)) in self._params:
                freq = self.get_parameter_value(param, (qubit,), None, True, group, cutoff_date)
            else:
                if meas_freq:
                    freq = self._backend.defaults().meas_freq_est[qubit]
                else:
                    freq = self._backend.defaults().qubit_freq_est[qubit]

            freqs.append(freq)

        return freqs

    def export_backend(self) -> Backend:
        """
        Exports the calibrations to a backend object that can be used.

        Returns:
            calibrated backend: A backend with the calibrations in it.
        """
        backend = copy.deepcopy(self._backend)

        backend.defaults().qubit_freq_est = self.get_frequencies(False)
        backend.defaults().meas_freq_est = self.get_frequencies(True)

        # TODO: build the instruction schedule map using the stored calibrations

        return backend

    @classmethod
    def from_csv(cls):
        """Create an instance from csv files"""
        raise NotImplementedError
