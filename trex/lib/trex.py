import logging
import time
from collections import OrderedDict
from copy import copy
from typing import overload, Iterable

import numpy as np
import pandas as pd
from rule_loe import RuleLoE
from sklearn.base import BaseEstimator, ClassifierMixin
from hdtree import AbstractSplitRule, TwentyQuantileSplit, TenQuantileSplit, FixedValueSplit, SingleCategorySplit, \
    TenQuantileRangeSplit, TwentyQuantileRangeSplit, SmallerThanSplit, HDTreeClassifier, GiniMeasure

from trex.lib.assignment import generate_rule_loe, generate_assignment_trees_iterator


class TrexClassifier(BaseEstimator, ClassifierMixin):
    allowed_splits: list[AbstractSplitRule]
    max_expert_depth: int
    min_samples_at_leaf: int
    max_assignment_depth: int
    max_experts: int
    feature_names: list[str] | None
    target_name: str | None
    pool_classifiers_: list[HDTreeClassifier]
    assignment_trees_: list[HDTreeClassifier]
    rule_loe_: RuleLoE | None = None
    min_rule_coverage: float
    min_rule_precision: float
    simplify_rules: bool
    _current_step: int
    _training_data_assignments: np.ndarray | None
    _X: np.ndarray | None
    _y: np.ndarray | None

    def __init__(self, allowed_splits: list[AbstractSplitRule] | None = None,
                 max_expert_depth: int = 3, min_samples_at_leaf: int = 3,
                 max_assignment_depth: int = 2, max_experts: int = 10,
                 feature_names: list[str] = None, target_name: str = None,
                 min_rule_coverage: float = 0.01, min_rule_precision: float = 0.5,
                 simplify_rules: bool = True,
                 random_state: int = 42
                 ):
        """
        :param allowed_splits: which rules are allowed to split the data
        :param max_expert_depth: how big the experts are allowed to grow
        :param min_samples_at_leaf: minimum sample size at leaf for an expert
        :param max_assignment_depth: Group separation (assignment) depth
        :param max_experts: maximum number of experts
        :param feature_names: list of feature names (must match the columns of the training data). if not given it will be extracted from the data frame for training. if training data is no dataframe, they will be enumerated
        :param target_name: name of the target column (must match the column of the training data). if not given it will be extracted from the data frame for training. if training data is a series, it will be named "target"
        :param min_rule_coverage: minimum coverage for a rule to be considered (otherwise discarded)
        :param min_rule_precision: minimum accuracy for a rule to be considered (otherwise discarded)
        :param simplify_rules: whether to simplify (prune horizontally) the rules after training
        :param random_state: random state for reproducibility (not used in the current implementation)
        """
        super().__init__()
        self.allowed_splits = allowed_splits or [
            TwentyQuantileSplit.build_with_restrictions(min_level=1),
            TwentyQuantileRangeSplit.build_with_restrictions(min_level=1),
            SmallerThanSplit.build_with_restrictions(min_level=1),
            SingleCategorySplit.build_with_restrictions(max_attributes=5),
            FixedValueSplit.build(),
            TenQuantileSplit.build(),
            TenQuantileRangeSplit.build(),
        ]

        self.max_expert_depth = max_expert_depth
        self.min_samples_at_leaf = min_samples_at_leaf
        self.max_assignment_depth = max_assignment_depth
        self.max_experts = max_experts
        self.feature_names = feature_names
        self.target_name = target_name
        self.min_rule_coverage = min_rule_coverage
        self.min_rule_precision = min_rule_precision
        self.simplify_rules = simplify_rules
        self._current_step = 0
        self.random_state = random_state
        self._validate_init_params()

    def _validate_init_params(self):
        if self.max_expert_depth < 1:
            raise ValueError("max_expert_depth must be greater than 0")

        if self.min_samples_at_leaf < 1:
            raise ValueError("min_samples_at_leaf must be greater than 0")

        if self.max_assignment_depth < 1:
            raise ValueError("max_assignment_depth must be greater than 0")

        if self.max_experts < 1:
            raise ValueError("max_experts must be greater than 0")

        if len(self.allowed_splits) == 0:
            raise ValueError("allowed_splits must not be empty")

    def _validate_fit_parameters(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series):
        if self.feature_names is None:
            if isinstance(X, pd.DataFrame):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f"f_{i}" for i in range(X.shape[1])]
        else:
            if len(self.feature_names) != X.shape[1]:
                raise ValueError(
                    f"Number of feature names ({len(self.feature_names)}) does not match number of features ({X.shape[1]})")

        if self.target_name is None:
            self.target_name = y.name if isinstance(y, pd.Series) else "target"

    def reset(self):
        """
        Resets the classifier to initial state before training
        :return:
        """
        self.pool_classifiers_ = []
        self.assignment_trees_ = []
        self.rule_loe_ = None
        self._current_step = 0
        self._training_data_assignments = None

    def create_assignment_labels(self) -> np.ndarray:
        return np.argmax(self._training_data_assignments, axis=1)

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series):
        self.reset()
        self._validate_fit_parameters(X, y)
        self._X = X if isinstance(X, np.ndarray) else X.values
        self._y = y if isinstance(y, np.ndarray) else y.values
        self._training_data_assignments = np.zeros((X.shape[0], self.max_experts), dtype=bool)
        total_time = 0
        for step in range(1, self.max_experts + 1):
            start = time.time()
            self.logger.info(f"Training step {step} of {self.max_experts}")
            self._current_step = step
            possible_models = self.emit_pool_expert_models()
            if not possible_models:
                raise Exception("No models emitted (Code: 34820938)")

            unassigned_mask = self._get_unassigned_mask()

            # go out if we assigned all samples
            if np.sum(unassigned_mask) == 0:
                self.logger.info(f"All data points are already assigned. Stopping early with {step - 1} experts.")
                break

            best_score = 0
            best_model = None
            for model in possible_models:
                model.fit(self._X[unassigned_mask], self._y[unassigned_mask])
                score = model.score(self._X[unassigned_mask], self._y[unassigned_mask])
                if score >= best_score:
                    best_score = score
                    best_model = model

            self.logger.debug(f"Best model for step {step} has score {best_score}")
            if step < self.max_experts:
                # find correctly classified samples and only assign the correctly classified ones to this expert
                predictions = best_model.predict(self._X[unassigned_mask])
                correctly_classified = predictions == self._y[unassigned_mask]

                indices = np.where(unassigned_mask)[0]
                self._training_data_assignments[indices[correctly_classified], step - 1] = True
            else:
                # last expert eats the rest
                self._training_data_assignments[:, step - 1] = unassigned_mask

            total_time += time.time() - start
            self.logger.debug(f"Training step {step} took {time.time() - start:.2f}s")
            self.pool_classifiers_.append(best_model)

        self.logger.debug(f"Training experts took {total_time:.2f}s")
        self.logger.info("Generating Rules")
        self.logger.info("Generate Assignment Trees")
        start = time.time()

        # simplify the trees if needed
        for tree in self.pool_classifiers_:
            if self.simplify_rules:
                tree.simplify(return_copy=False)

        # generate and optionally simplify the assignment trees.
        # could be parallelized
        for tree in generate_assignment_trees_iterator(self):
            self.logger.debug(f"Generated assignment tree #{len(self.assignment_trees_) + 1}")
            if self.simplify_rules:
                tree.simplify(return_copy=False)

            self.assignment_trees_.append(tree)

        self.rule_loe_ = generate_rule_loe(self)
        self.logger.debug(f"Generating rules took {time.time() - start:.2f}s")

    def _get_unassigned_mask(self) -> np.ndarray:
        return np.sum(self._training_data_assignments, axis=1) == 0

    @property
    def logger(self) -> logging.Logger:
        return logging.getLogger(__class__.__name__)

    def emit_pool_expert_models(self) -> list[HDTreeClassifier]:
        """
        Emits models which can be used for next training step
        The best working model will be selected
        :return:
        """
        return [HDTreeClassifier(
            attribute_names=self.feature_names,
            max_levels=self.max_expert_depth,
            min_samples_at_leaf=self.min_samples_at_leaf,
            information_measure=GiniMeasure(),
            allowed_splits=[copy(split) for split in self.allowed_splits],
        )]

    def emit_assignment_model(self, expert_number: int) -> HDTreeClassifier:
        return HDTreeClassifier(
            attribute_names=self.feature_names,
            max_levels=self.max_assignment_depth,
            min_samples_at_leaf=self.min_samples_at_leaf,
            information_measure=GiniMeasure(),
            allowed_splits=[copy(split) for split in self.allowed_splits],
        )

    def explain_datapoints(self, data_points: Iterable[pd.Series | np.ndarray | list] | pd.DataFrame | np.ndarray | pd.Series) -> pd.DataFrame:
        entries = []
        if isinstance(data_points, pd.DataFrame):
            iterator = data_points.iterrows()
        elif isinstance(data_points, pd.Series):
            iterator = enumerate([data_points])

        elif isinstance(data_points, np.ndarray):
            iterator = enumerate(data_points)

        else:
            iterator = enumerate(data_points)

        for index, data_point in iterator:
            if isinstance(data_point, pd.Series):
                data_point = data_point.values

            if isinstance(data_point, list):
                data_point = np.array(data_point, dtype="O")

            # check shape must be exactly one sample
            if len(data_point.shape) != 1:
                raise ValueError("each data point must be a single 1D sample")

            concept_rule_clause = self.rule_loe_.get_best_fitting_rule(data_point)
            if concept_rule_clause is None:
                entries.append({})

            else:
                concept, rg = concept_rule_clause
                entries.append({
                    **{f"Rule {i + 1}": str(rule) for i, rule in enumerate(concept.readable_rules)},
                    "Precision in %": f"{concept.precision * 100:.2f}",
                    "Coverage in %": f"{concept.coverage * 100:.2f}",
                    "Expert": concept.nerd_idx + 1,
                    f"{self.target_name}": concept.target_concept,
                })

        max_rule_length = max([len([key for key in entry if key.startswith("Rule ")]) for entry in entries])

        return pd.DataFrame(
            columns=[f"Rule {i + 1}" for i in range(max_rule_length)] + ["Precision in %", "Coverage in %", "Expert", self.target_name],
            data=entries).fillna("")

    def explain(self, only_for_concept: str | int | None = None):
        return self.rule_loe_.explain(only_for_concept)

    def get_X(self) -> np.ndarray:
        return self._X

    def get_y(self) -> np.ndarray:
        return self._y

    def predict(self, X):
        return self.rule_loe_.predict(X)
