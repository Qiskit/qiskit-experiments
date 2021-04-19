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

"""Class to store and manage the results of a calibration experiments."""

import dataclasses
from collections import namedtuple, defaultdict
from datetime import datetime
from typing import Any, Dict, Set, Tuple, Union, List, Optional

from qiskit.providers.backend import BackendV1 as Backend
from qiskit.circuit import Gate
from qiskit import QuantumCircuit
from qiskit.pulse import Schedule, DriveChannel, ControlChannel, MeasureChannel
from qiskit.pulse.channels import PulseChannel
from qiskit.circuit import Parameter
from qiskit_experiments.calibration.exceptions import CalibrationError
from qiskit_experiments.calibration.parameter_value import ParameterValue

ParameterKey = namedtuple("ParameterKey", ["schedule", "parameter", "qubits"])


class Calibrations:
    """
    A class to manage schedules with calibrated parameter values.
    Schedules are stored in a dict and are intended to be fully parameterized,
    including the index of the channels. The parameter values are stored in a
    dict where parameters are keys. This class supports:
    - having different schedules share parameters
    - allows default schedules for qubits that can be overridden of specific qubits.
    """

    def __init__(self, backend: Backend):
        """
        Initialize the instructions from a given backend.

        Args:
            backend: The backend from which to get the configuration.
        """

        self._n_qubits = backend.configuration().num_qubits
        self._n_uchannels = backend.configuration().n_uchannels
        self._config = backend.configuration()

        # Dict of the form: (schedule.name, parameter.name, qubits): Parameter
        self._parameter_map = {}

        # Default dict of the form: (schedule.name, parameter.name, qubits): [ParameterValue, ...]
        self._params = defaultdict(list)

        self._schedules = {}

    def add_schedule(self, schedule: Schedule, qubits: Tuple = None):
        """
        Add a schedule and register its parameters.

        Args:
            schedule: The schedule to add.
            qubits: The qubits for which to add the schedules. If None is given then this
                schedule is the default schedule for all qubits.

        Raises:
            CalibrationError: If the parameterized channel index is not formatted
                following index1.index2... or if several parameters in the same schedule
                have the same name.
        """
        # check that channels, if parameterized, have the proper name format.
        # pylint: disable = raise-missing-from
        param_indices = set()
        for ch in schedule.channels:
            if isinstance(ch.index, Parameter):
                param_indices.add(ch.index)
                try:
                    [int(index) for index in ch.index.name.split(".")]
                except ValueError:
                    raise CalibrationError(
                        "Parameterized channel must have a name "
                        "formatted following index1.index2..."
                    )

        self._schedules[(schedule.name, qubits)] = schedule

        param_names = [param.name for param in schedule.parameters]

        if len(param_names) != len(set(param_names)):
            raise CalibrationError(f"Parameter names in {schedule.name} must be unique.")

        # Register parameters that are not indices.
        for param in schedule.parameters:
            if param not in param_indices:
                self.register_parameter(param, schedule, qubits)

    def register_parameter(
        self, parameter: Parameter, schedule: Schedule = None, qubits: Tuple = None
    ):
        """
        Registers a parameter for the given schedule. This allows self to determine the
        parameter instance that corresponds to the given schedule name, parameter name
        and qubits.

        Args:
            parameter: The parameter to register.
            schedule: The Schedule to which this parameter belongs. The schedule can
                be None which allows the calibration to accommodate, e.g. qubit frequencies.
            qubits: The qubits for which to register the parameter.
        """
        sched_name = schedule.name if schedule else None
        self._parameter_map[ParameterKey(sched_name, parameter.name, qubits)] = parameter

    @property
    def parameters(self) -> Dict[Parameter, Set]:
        """
        Returns a dictionary mapping parameters managed by the calibrations to the schedules and
        qubits using the parameters. The values of the dict are sets containing the names of the
        schedules and qubits in which the parameter appears. Parameters that are not attached to
        a schedule will have None in place of a schedule name.
        """
        parameters = defaultdict(set)
        for key, param in self._parameter_map.items():
            schedule_name = key.schedule

            if key.qubits:
                parameters[param].add((schedule_name, key.qubits))
            else:
                parameters[param].add((schedule_name,))

        return parameters

    def add_parameter_value(
        self,
        value: ParameterValue,
        param: Union[Parameter, str],
        qubits: Tuple[int, ...] = None,
        schedule: Union[Schedule, str] = None,
    ):
        """
        Add a parameter value to the stored parameters. This parameter value may be
        applied to several channels, for instance, all DRAG pulses may have the same
        standard deviation.

        Args:
            value: The value of the parameter to add.
            param: The parameter or its name for which to add the measured value.
            qubits: The qubits to which this parameter applies.
            schedule: The schedule or its name for which to add the measured parameter value.

        Raises:
            CalibrationError: if ch_type is not given when chs are None, if the
                channel type is not a ControlChannel, DriveChannel, or MeasureChannel, or
                if the parameter name is not already in self.
        """
        param_name = param.name if isinstance(param, Parameter) else param
        sched_name = schedule.name if isinstance(schedule, Schedule) else schedule

        # First look for a parameter that matches the given qubits.
        if (sched_name, param_name, qubits) in self._parameter_map:
            param = self._parameter_map[ParameterKey(sched_name, param_name, qubits)]

        # If no parameter was found look for a default parameter
        else:
            param = self._parameter_map[ParameterKey(sched_name, param_name, None)]

        if param is None:
            raise CalibrationError(
                f"No parameter found for parameter {param_name} in "
                f"schedule {sched_name} and qubits {qubits}."
            )

        # Find all schedules that share this parameter
        common_schedules = {ParameterKey(sched_name, param_name, qubits)}
        for key in self._parameter_map.keys():
            if self._parameter_map[key] == param:
                common_schedules.add(key)

        for key in common_schedules:
            self._params[key].append(value)

    def _get_channel_index(self, qubits: Tuple, chan: PulseChannel, control_index: int = 0) -> int:
        """
        Get the index of the parameterized channel based on the given qubits
        and the name of the parameter in the channel index. The name of this
        parameter for control channels must be written as qubit_index1.qubit_index2... .
        For example, the following parameter names are valid: '1', '1.0', '30.12'.

        Args:
            qubits: The qubits for which we want to obtain the channel index.
            chan: The channel with a parameterized name.
            control_index: An index used to specify which control channel to use if a given
            pair of qubits has more than one control channel.

        Returns:
            index: The index of the channel. For example, if qubits=(10, 32) and
                chan is a control channel with parameterized index name '1.0'
                the method returns the control channel corresponding to
                qubits (qubits[1], qubits[0]) which is here the control channel of
                qubits (32, 10).

        Raises:
            CalibrationError: if the number of qubits is incorrect, if the
                number of inferred ControlChannels is not correct, or if ch is not
                a DriveChannel, MeasureChannel, or ControlChannel.
        """

        if isinstance(chan.index, Parameter):
            if isinstance(chan, (DriveChannel, MeasureChannel)):
                if len(qubits) != 1:
                    raise CalibrationError(f"Too many qubits given for {chan.__class__.__name__}.")

                return qubits[0]

            if isinstance(chan, ControlChannel):
                indices = [int(sub_channel) for sub_channel in chan.index.name.split(".")]
                ch_qubits = tuple(qubits[index] for index in indices)
                chs_ = self._config.control(ch_qubits)

                if len(chs_) < control_index:
                    raise CalibrationError(
                        f"Control channel index {control_index} not found for qubits {qubits}."
                    )

                return chs_[control_index].index

            raise CalibrationError(f"{chan} must be a sub-type of {PulseChannel}.")

        return chan.index

    def get_parameter_value(
        self,
        param: Union[Parameter, str],
        qubits: Tuple[int, ...],
        schedule: Union[Schedule, str, None] = None,
        valid_only: bool = True,
        group: str = "default",
        cutoff_date: datetime = None,
    ) -> Union[int, float, complex]:
        """
        1) Check if the given qubits have their own Parameter.
        2) If they do not check to see if a parameter global to all qubits exists.
        3) Filter candidate parameter values.
        4) Return the most recent parameter.

        Args:
            param: The parameter or the name of the parameter for which to get the parameter value.
            qubits: The qubits for which to get the value of the parameter.
            schedule: The schedule or its name for which to get the parameter value.
            valid_only: Use only parameters marked as valid.
            group: The calibration group from which to draw the
                parameters. If not specifies this defaults to the 'default' group.
            cutoff_date: Retrieve the most recent parameter up until the cutoff date. Parameters
                generated after the cutoff date will be ignored. If the cutoff_date is None then
                all parameters are considered. This allows users to discard more recent values that
                may be erroneous.

        Returns:
            value: The value of the parameter.

        Raises:
            CalibrationError: if there is no parameter value for the given parameter name
                and pulse channel.
        """
        param_name = param.name if isinstance(param, Parameter) else param
        sched_name = schedule.name if isinstance(schedule, Schedule) else schedule

        # 1) Check for qubit specific parameters.
        if (sched_name, param_name, qubits) in self._params:
            candidates = self._params[(sched_name, param_name, qubits)]

        # 2) Check for default values.
        elif (sched_name, param_name, None) in self._params:
            candidates = self._params[(sched_name, param_name, None)]
        else:
            raise CalibrationError(
                f"No parameter for {param_name} and schedule {sched_name} "
                f"and qubits {qubits}. No default value exists."
            )

        # 3) Filter candidate parameter values.
        if valid_only:
            candidates = [val for val in candidates if val.valid]

        candidates = [val for val in candidates if val.group == group]

        if cutoff_date:
            candidates = [val for val in candidates if val.date_time <= cutoff_date]

        if len(candidates) == 0:
            msg = f"No candidate parameter values for {param_name} in calibration group {group} "

            if qubits:
                msg += f"on qubits {qubits} "

            msg += f"in schedule {sched_name}"

            if cutoff_date:
                msg += f" Cutoff date: {cutoff_date}"

            raise CalibrationError(msg)

        # 4) Return the most recent parameter.
        candidates.sort(key=lambda x: x.date_time)

        return candidates[-1].value

    def get_schedule(
        self,
        name: str,
        qubits: Tuple[int, ...],
        free_params: List[str] = None,
        group: Optional[str] = "default",
    ) -> Schedule:
        """
        Get the schedule with the non-free parameters assigned to their values.

        Args:
            name: The name of the schedule to get.
            qubits: The qubits for which to get the schedule.
            free_params: The parameters that should remain unassigned.
            group: The calibration group from which to draw the
                parameters. If not specifies this defaults to the 'default' group.

        Returns:
            schedule: A copy of the template schedule with all parameters assigned
                except for those specified by free_params.

        Raises:
            CalibrationError: if the name of the schedule is not known.
        """
        if (name, qubits) in self._schedules:
            schedule = self._schedules[(name, qubits)]
        elif (name, None) in self._schedules:
            schedule = self._schedules[(name, None)]
        else:
            raise CalibrationError(f"Schedule {name} is not defined for qubits {qubits}.")

        # Retrieve the channel indices based on the qubits and bind them.
        binding_dict = {}
        for ch in schedule.channels:
            if ch.is_parameterized():
                binding_dict[ch.index] = self._get_channel_index(qubits, ch)

        # Loop through the remaining parameters in the schedule, get their values and bind.
        if free_params is None:
            free_params = []

        for param in schedule.parameters:
            if param.name not in free_params and param not in binding_dict:
                binding_dict[param] = self.get_parameter_value(
                    param.name, qubits, name, group=group
                )

        return schedule.assign_parameters(binding_dict, inplace=False)

    def get_circuit(
        self,
        schedule_name: str,
        qubits: Tuple,
        free_params: List[str] = None,
        group: Optional[str] = "default",
        schedule: Schedule = None,
    ) -> QuantumCircuit:
        """
        Queries a schedule by name for the given set of qubits. The parameters given
        under the list free_params are left unassigned. The queried schedule is then
        embedded in a gate with a calibration and returned as a quantum circuit.

        Args:
            schedule_name: The name of the schedule to retrieve.
            qubits: The qubits for which to generate the gate with the schedule in it.
            free_params: Names of the parameters that will remain unassigned.
            group: The calibration group from which to retrieve the calibrated values.
                If unspecified this defaults to 'default'.
            schedule: The schedule to add to the gate if the internally stored one is
                not used.

        Returns:
            A quantum circuit in which the parameter values have been assigned aside from
            those explicitly specified in free_params.
        """
        if schedule is None:
            schedule = self.get_schedule(schedule_name, qubits, free_params, group)

        gate = Gate(name=schedule_name, num_qubits=len(qubits), params=list(schedule.parameters))
        circ = QuantumCircuit(len(qubits), len(qubits))
        circ.append(gate, list(range(len(qubits))))
        circ.add_calibration(gate, qubits, schedule, params=schedule.parameters)

        return circ

    def schedules(self) -> List[Dict[str, Any]]:
        """
        Return the schedules in self in a list of dictionaries to help
        users manage their schedules.

        Returns:
            data: A list of dictionaries with all the schedules in it.
        """
        data = []
        for context, sched in self._schedules.items():
            data.append({"context": context, "schedule": sched, "parameters": sched.parameters})

        return data

    def parameters_table(
        self,
        parameters: List[str] = None,
        schedules: List[Union[Schedule, str]] = None,
        qubit_list: List[Tuple[int, ...]] = None,
    ) -> List[Dict[str, Any]]:
        """

        Args:
            parameters: The parameter names that should be included in the returned
                table. If None is given then all names are included.
            schedules: The schedules to which to restrict the output.
            qubit_list: The qubits that should be included in the returned table.
                If None is given then all channels are returned.

        Returns:
            data: A dictionary of parameter values which can easily be converted to a
                data frame.
        """

        data = []

        # Convert inputs to lists of strings
        if parameters is not None:
            parameters = {prm.name if isinstance(prm, Parameter) else prm for prm in parameters}

        if schedules is not None:
            schedules = {sdl.name if isinstance(sdl, Schedule) else sdl for sdl in schedules}

        # Look for exact matches. Default values will be ignored.
        keys = set()
        for key, param in self._params.items():
            if parameters and key.parameter not in parameters:
                continue
            if schedules and key.schedule not in schedules:
                continue
            if qubit_list and key.qubits not in qubit_list:
                continue

            keys.add(key)

        for key in keys:
            for value in self._params[key]:
                value_dict = dataclasses.asdict(value)
                value_dict["qubits"] = key.qubits
                value_dict["parameter"] = key.parameter
                value_dict["schedule"] = key.schedule

                data.append(value_dict)

        return data

    def to_db(self):
        """
        Serializes the parameterized schedules and parameter values so
        that they can be sent and stored in an external DB.
        """
        raise NotImplementedError

    def from_db(self):
        """
        Retrieves the parameterized schedules and pulse parameters from an
        external DB.
        """
        raise NotImplementedError
