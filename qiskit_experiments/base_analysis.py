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
Base analysis class.
"""

from abc import ABC, abstractmethod
import copy

from qiskit.providers.options import Options
from qiskit.exceptions import QiskitError
from .experiment_data import ExperimentData, AnalysisResult


class BaseAnalysis(ABC):
    """Base Analysis class for analyzing Experiment data."""

    # Expected experiment data container for analysis
    __experiment_data__ = ExperimentData

    def __init__(self, **options):
        """Initialize a base analysis class

        Args:
            options: kwarg options for analysis.
        """
        self._options = self._default_options()
        self.set_options(**options)

    @classmethod
    def _default_options(cls):
        return Options()

    def set_options(self, **fields):
        """Set the analysis options.

        Args:
            fields: The fields to update the options
        """
        self._options.update_options(**fields)

    @property
    def options(self):
        """Return the analysis options.

        The options of an analysis class are used to provide kwarg values for
        the :meth:`run` method.
        """
        return self._options

    def run(self, experiment_data, save=True, return_figures=False, **options):
        """Run analysis and update stored ExperimentData with analysis result.

        Args:
            experiment_data (ExperimentData): the experiment data to analyze.
            save (bool): if True save analysis results and figures to the
                         :class:`ExperimentData`.
            return_figures (bool): if true return a pair of
                                   ``(analysis_results, figures)``,
                                    otherwise return only analysis_results.
            options: additional analysis options. Any values set here will
                     override the value from :meth:`options` for the current run.

        Returns:
            AnalysisResult: the output of the analysis that produces a
                            single result.
            List[AnalysisResult]: the output for analysis that produces
                                  multiple results.
            Tuple: If ``return_figures=True`` the output is a pair
                   ``(analysis_results, figures)`` where  ``analysis_results``
                   may be a single or list of :class:`AnalysisResult` objects, and
                   ``figures`` may be None, a single figure, or a list of figures.

        Raises:
            QiskitError: if experiment_data container is not valid for analysis.
        """
        if not isinstance(experiment_data, self.__experiment_data__):
            raise QiskitError(
                f"Invalid experiment data type, expected {self.__experiment_data__.__name__}"
                f" but received {type(experiment_data).__name__}"
            )

        # Wait for experiment job to finish
        # experiment_data.block_for_result()

        # Get runtime analysis options
        analysis_options = copy.copy(self.options)
        analysis_options.update_options(**options)
        analysis_options = analysis_options.__dict__

        # Run analysis
        analysis_results, figures = self._run_analysis(experiment_data, **analysis_options)

        # Save to experiment data
        if save:
            if isinstance(analysis_results, AnalysisResult):
                experiment_data.add_analysis_result(analysis_results)
            else:
                for res in analysis_results:
                    experiment_data.add_analysis_result(res)
            if figures:
                for fig in figures:
                    experiment_data.add_figure(fig)
        if return_figures:
            return analysis_results, figures
        return analysis_results

    @abstractmethod
    def _run_analysis(self, experiment_data, **options):
        """Run analysis on circuit data.

        Args:
            experiment_data (ExperimentData): the experiment data to analyze.
            options: additional options for analysis. By default the fields and
                     values in :meth:`options` are used and any provided values
                     can override these.

        Returns:
            tuple: A pair ``(analysis_results, figures)`` where
                   ``analysis_results`` may be a single or list of
                   AnalysisResult objects, and ``figures`` may be
                   None, a single figure, or a list of figures.
        """
        pass
