from __future__ import annotations
import typing

if typing.TYPE_CHECKING:
    from .trex import TrexClassifier

from typing import Iterable

import numpy as np
import rule_loe
from hdtree import HDTreeClassifier, Node, simplify_rules
from rule_loe import LABEL_EXPERT, LABEL_NO_EXPERT, Concept


def generate_assignment_trees_iterator(model: TrexClassifier, ) -> Iterable[HDTreeClassifier]:
    """
    Generates decision trees that learns the assignment function from data point to model

    :return:
    """

    all_data = model.get_X()
    assignments = model.create_assignment_labels()
    max_val = np.max(assignments)

    for model_idx in range(max_val+1):
        ass_tree = model.emit_assignment_model(expert_number=model_idx + 1)
        ass_tree.fit(all_data, np.where(assignments == model_idx,
                                        LABEL_EXPERT,
                                        LABEL_NO_EXPERT))

        yield ass_tree


def generate_rule_loe(trex: TrexClassifier):
    trees_assignment = trex.assignment_trees_


    concepts = []
    experts = [expert if trex.simplify_rules is False else expert.simplify(return_copy=True) for expert in
               trex.pool_classifiers_]

    for tree_index, ass_tree in enumerate(trees_assignment):
        nerd_tree: HDTreeClassifier = experts[tree_index]
        agreeing_sample_indices_assignment_all_data: list[int] = []
        assignment_conditions_nodes: list[list[Node]] = []

        # nerd_conditions_nodes: List[List[Node]] = []

        # assignment rules
        leafs_assignment: list[Node] = [node for node in ass_tree.get_all_nodes_below_node(node=None) if node.is_leaf()]

        if len(leafs_assignment) == 0:
            leafs_assignment = [ass_tree.get_head()]

        for leaf_nerd in leafs_assignment:
            concept = ass_tree.get_prediction_for_node(leaf_nerd)  # the actual class

            if concept == LABEL_EXPERT:
                chain = ass_tree.follow_node_to_root(node=leaf_nerd) + [leaf_nerd]
                agreeing_sample_indices_assignment_all_data += list(leaf_nerd.get_data_indices())
                assignment_conditions_nodes.append(chain)

        # nerd rules
        leafs_nerd = [node for node in nerd_tree.get_all_nodes_below_node(node=None) if node.is_leaf()]

        if len(leafs_nerd) == 0:
            leafs_nerd = [nerd_tree.get_head()]

        # follow each experts' leaf and cross join with each assignment trees' path to expert
        for leaf_nerd in leafs_nerd:
            # get the leafs decision
            concept = nerd_tree.get_prediction_for_node(node=leaf_nerd)
            concept_readable = concept  # loe.enc_.inverse_transform([concept])[0]
            nerd_chain = nerd_tree.follow_node_to_root(leaf_nerd) + [leaf_nerd]

            # now check each possible way that lands at that expert
            for assignment_option_idx in range(0, len(assignment_conditions_nodes)):

                # get all samples withing current assignment node
                assignment_leaf_node = assignment_conditions_nodes[assignment_option_idx][-1]
                assignment_sample = trex.get_X()[assignment_leaf_node.get_data_indices()]
                assignment_targets = trex.get_y()[assignment_leaf_node.get_data_indices()]

                # get all samples that are in assignment model AND follow expert path to current leaf
                flow_samples_mask = [nerd_tree.extract_node_chain_for_sample(dp)[-1] == leaf_nerd for dp in
                                     assignment_sample]

                if sum(flow_samples_mask) == 0:
                    continue

                # same_flow = assignment_sample[flow_samples_mask]
                y_sample = assignment_targets[flow_samples_mask].astype(str)
                prec = sum(y_sample == [str(concept)]) / sum(flow_samples_mask)
                cov = sum(flow_samples_mask) / len(trex.get_X())

                # option_ass_conditions_nodes = assignment_conditions_nodes[assignment_option_idx]
                nodes_complete = assignment_conditions_nodes[assignment_option_idx][:-1] + nerd_chain[:-1]
                rules_complete = [node.get_split_rule() for node in nodes_complete if node.get_split_rule() is not None]

                data_point = trex.get_X()[assignment_leaf_node.get_data_indices()][flow_samples_mask][0]
                rules_simplified = simplify_rules(rules=rules_complete,
                                                  model_to_sample={nerd_tree: data_point,
                                                                   ass_tree: data_point}
                                                  )

                chain = nerd_tree.follow_node_to_root(node=leaf_nerd)
                # nerd_conditions_nodes.append(chain)

                readable_rules = [
                    rule.explain_split(sample=data_point, hide_sample_specifics=True)
                    for rule in rules_simplified]

                # gather expected rule outcomes assignment trees on sample
                is_nerd_tree = [rule.get_tree() is nerd_tree for rule in rules_simplified]
                nodes_simplified = [rule.get_node() for rule in rules_simplified
                                    if rule.get_node().get_split_rule() is not None]
                concept_description = Concept(
                    target_concept=concept_readable,
                    nodes_complete=nodes_complete,
                    nodes_expert=chain,
                    nodes_assignment=assignment_conditions_nodes[assignment_option_idx],
                    nodes_simplified=nodes_simplified,
                    simplified_node_came_from_nerd_tree=is_nerd_tree,
                    readable_rules=readable_rules,
                    precision=prec,
                    coverage=cov,
                    simplified_tree=trex.simplify_rules,
                    nerd_idx=tree_index,
                    assignment_feature_mask=[True for _ in range(len(trex.get_X()[0]))],
                    sample_dummy=list(data_point),
                    original_attribute_names=trex.feature_names,
                )

                concepts.append(concept_description)

    return rule_loe.RuleLoE(concepts=concepts, min_precision=trex.min_rule_precision,
                            min_coverage=trex.min_rule_coverage)
