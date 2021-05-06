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
Base Experiment class.
"""

from abc import ABC, abstractmethod
from numbers import Integral

from qiskit import transpile, assemble
from qiskit.exceptions import QiskitError

from qiskit.providers.experiment import ExperimentDataV1


_TRANSPILE_OPTIONS = {
    "basis_gates",
    "coupling_map",
    "backend_properties",
    "initial_layout",
    "layout_method",
    "routing_method",
    "translation_method",
    "scheduling_method",
    "instruction_durations",
    "dt",
    "seed_transpiler",
    "optimization_level",
    "pass_manager",
    "callback",
    "output_name",
}


class BaseExperiment(ABC):
    """Base Experiment class

    Class Attributes:

        __analysis_class__: Optional, the default Analysis class to use for
                            data analysis. If None no data analysis will be
                            done on experiment data (Default: None).
        __experiment_data__: ExperimentData class that is produced by the
                             experiment (Default: ExperimentData).
    """

    # Analysis class for experiment
    __analysis_class__ = None

    # ExperimentData class for experiment
    __experiment_data__ = ExperimentDataV1

    # Custom default transpiler options for experiment subclasses
    __transpile_defaults__ = {"optimization_level": 0}

    # Custom default run (assemble) options for experiment subclasses
    __run_defaults__ = {}

    def __init__(self, qubits, experiment_type=None, circuit_options=None):
        """Initialize the experiment object.

        Args:
            qubits (int or Iterable[int]): the number of qubits or list of
                                           physical qubits for the experiment.
            experiment_type (str): Optional, the experiment type string.
            circuit_options (Iterable): Optional, list of kwarg names for
                                        the subclassed `circuit` method.

        Raises:
            QiskitError: if qubits is a list and contains duplicates.
        """
        # Experiment identification metadata
        self._type = experiment_type if experiment_type else type(self).__name__

        # Circuit parameters
        if isinstance(qubits, Integral):
            self._num_qubits = qubits
            self._physical_qubits = tuple(range(qubits))
        else:
            self._num_qubits = len(qubits)
            self._physical_qubits = tuple(qubits)
            if self._num_qubits != len(set(self._physical_qubits)):
                raise QiskitError("Duplicate qubits in physical qubits list.")

        # Store options and values
        self._circuit_options = set(circuit_options) if circuit_options else set()

    def run(self, backend, experiment_data=None, **kwargs):
        """Run an experiment and perform analysis.

        Args:
            backend (Backend): The backend to run the experiment on.
            experiment_data (ExperimentData): Optional, add results to existing
                experiment data. If None a new ExperimentData object will be
                returned.
            kwargs: keyword arguments for self.circuit,
                    qiskit.transpile, and backend.run.

        Returns:
            ExperimentData: the experiment data object.
        """
        # NOTE: This method is intended to be overriden by subclasses if required.

        # Create new experiment data
        if experiment_data is None:
            experiment_data = self.__experiment_data__(backend, self._type)

        # Filter kwargs
        run_options = self.__run_defaults__.copy()
        circuit_options = {}
        for key, value in kwargs.items():
            if key in _TRANSPILE_OPTIONS or key in self._circuit_options:
                circuit_options[key] = value
            else:
                run_options[key] = value

        # Generate and run circuits
        circuits = self.transpiled_circuits(backend, **circuit_options)
        qobj = assemble(circuits, backend, **run_options)
        job = backend.run(qobj)

        # Add Job to ExperimentData
        experiment_data.add_data(job)

        # Queue analysis of data for when job is finished
        if self.__analysis_class__ is not None:
            # pylint: disable = not-callable
            self.__analysis_class__().run(experiment_data, **kwargs)

        # Return the ExperimentData future
        return experiment_data

    @property
    def num_qubits(self):
        """Return the number of qubits for this experiment."""
        return self._num_qubits

    @property
    def physical_qubits(self):
        """Return the physical qubits for this experiment."""
        return self._physical_qubits

    @classmethod
    def analysis(cls, **kwargs):
        """Return the default Analysis class for the experiment."""
        if cls.__analysis_class__ is None:
            raise QiskitError(
                f"Experiment {cls.__name__} does not define" " a default Analysis class"
            )
        # pylint: disable = not-callable
        return cls.__analysis_class__(**kwargs)

    @abstractmethod
    def circuits(self, backend=None, **circuit_options):
        """Return a list of experiment circuits.

        Args:
            backend (Backend): Optional, a backend object.
            circuit_options: kwarg options for the function.

        Returns:
            List[QuantumCircuit]: A list of :class:`QuantumCircuit`s.

        .. note:
            These circuits should be on qubits ``[0, .., N-1]`` for an
            *N*-qubit experiment. The circuits mapped to physical qubits
            are obtained via the :meth:`transpiled_circuits` method.
        """
        # NOTE: Subclasses should override this method with explicit
        # kwargs for any circuit options rather than use `**circuit_options`.
        # This allows these options to have default values, and be
        # documented in the methods docstring for the API docs.

    def transpiled_circuits(self, backend=None, **kwargs):
        """Return a list of experiment circuits.

        Args:
            backend (Backend): Optional, a backend object to use as the
                               argument for the :func:`qiskit.transpile`
                               function.
            kwargs: kwarg options for the :meth:`circuits` method, and
                    :func:`qiskit.transpile` function.

        Returns:
            List[QuantumCircuit]: A list of :class:`QuantumCircuit`s.

        Raises:
            QiskitError: if an initial layout is specified in the
                         kwarg options for transpilation. The initial
                         layout must be generated from the experiment.

        .. note:
            These circuits should be on qubits ``[0, .., N-1]`` for an
            *N*-qubit experiment. The circuits mapped to physical qubits
            are obtained via the :meth:`transpiled_circuits` method.
        """
        # Filter kwargs to circuit and transpile options
        circuit_options = {}
        transpile_options = self.__transpile_defaults__.copy()
        for key, value in kwargs.items():
            valid_key = False
            if key in self._circuit_options:
                circuit_options[key] = value
                valid_key = True
            if key in _TRANSPILE_OPTIONS:
                transpile_options[key] = value
                valid_key = True
            if not valid_key:
                raise QiskitError(
                    f"{key} is not a valid kwarg for" f" {self.circuits} or {transpile}"
                )

        # Generate circuits
        circuits = self.circuits(backend=backend, **circuit_options)

        # Transpile circuits
        if "initial_layout" in transpile_options:
            raise QiskitError("Initial layout must be specified by the Experiement.")
        transpile_options["initial_layout"] = self.physical_qubits
        circuits = transpile(circuits, backend=backend, **transpile_options)

        return circuits
