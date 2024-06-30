# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# pylint: disable=import-error
"""
MindNLP Graphormer data collator
"""


from typing import Mapping

import numpy as np

from ....utils import is_cython_available, requires_backends

if is_cython_available():
    import pyximport

    pyximport.install(setup_args={"include_dirs": np.get_include()})
    from . import algos_graphormer  # pylint: disable=no-name-in-module


def convert_to_single_emb(node_feature, offset: int = 512):
    """Convert to single embedding"""
    feature_num = node_feature.shape[1] if len(node_feature.shape) > 1 else 1
    feature_offset = 1 + np.arange(0, feature_num * offset, offset, dtype=np.int64)
    node_feature = node_feature + feature_offset
    return node_feature


def preprocess_item(item, keep_features=True):
    """Process each item of the graph dataset"""
    requires_backends(preprocess_item, ["cython"])

    if keep_features and "edge_attr" in item.keys():  # edge_attr
        edge_attr = np.asarray(item["edge_attr"], dtype=np.int64)
    else:
        edge_attr = np.ones((len(item["edge_index"][0]), 1), dtype=np.int64)  # same embedding for all

    if keep_features and "node_feat" in item.keys():  # input_nodes
        node_feature = np.asarray(item["node_feat"], dtype=np.int64)
    else:
        node_feature = np.ones((item["num_nodes"], 1), dtype=np.int64)  # same embedding for all

    edge_index = np.asarray(item["edge_index"], dtype=np.int64)

    input_nodes = convert_to_single_emb(node_feature) + 1
    num_nodes = item["num_nodes"]

    if len(edge_attr.shape) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = np.zeros([num_nodes, num_nodes, edge_attr.shape[-1]], dtype=np.int64)
    attn_edge_type[edge_index[0], edge_index[1]] = convert_to_single_emb(edge_attr) + 1

    # node adj matrix [num_nodes, num_nodes] bool
    adj = np.zeros([num_nodes, num_nodes], dtype=bool)
    adj[edge_index[0], edge_index[1]] = True

    shortest_path_result, path = algos_graphormer.floyd_warshall(adj)
    max_dist = np.amax(shortest_path_result)

    input_edges = algos_graphormer.gen_edge_input(max_dist, path, attn_edge_type)
    attn_bias = np.zeros([num_nodes + 1, num_nodes + 1], dtype=np.single)  # with graph token

    # combine
    item["input_nodes"] = input_nodes + 1  # we shift all indices by one for padding
    item["attn_bias"] = attn_bias
    item["attn_edge_type"] = attn_edge_type
    item["spatial_pos"] = shortest_path_result.astype(np.int64) + 1  # we shift all indices by one for padding
    item["in_degree"] = np.sum(adj, axis=1).reshape(-1) + 1  # we shift all indices by one for padding
    item["out_degree"] = item["in_degree"]  # for undirected graph
    item["input_edges"] = input_edges + 1  # we shift all indices by one for padding
    if "labels" not in item:
        item["labels"] = item["y"]

    return item


class GraphormerDataCollator:
    """
    Graphormer data collator

    Converts graph dataset into the format accepted by Graphormer model
    """
    def __init__(self, spatial_pos_max=20, on_the_fly_processing=False):
        """
        Initializes a new instance of the GraphormerDataCollator class.
        
        Args:
            self: The object instance.
            spatial_pos_max (int): The maximum spatial position value. Defaults to 20.
            on_the_fly_processing (bool): Indicates whether on-the-fly processing is enabled or not. Defaults to False.
        
        Returns:
            None.
        
        Raises:
            ImportError: If the required Cython package (pyximport) is not available.
        
        """
        if not is_cython_available():
            raise ImportError("Graphormer preprocessing needs Cython (pyximport)")

        self.spatial_pos_max = spatial_pos_max
        self.on_the_fly_processing = on_the_fly_processing
        self.output_columns=["attn_bias",
                             "attn_edge_type",
                             "spatial_pos",
                             "in_degree",
                             "input_nodes",
                             "input_edges",
                             "out_degree",
                             "labels"]

    def __call__(self, edge_index, edge_attr, y, num_nodes, node_feat, batch_info):
        """
        This method, named '__call__', is defined within the class 'GraphormerDataCollator' and is used to process data
        for graph neural network models. It takes the following parameters:
        
        Args:
            self: The instance of the class.
            edge_index (List): A list of edge indices representing the connectivity of nodes in the graph.
            edge_attr (List): A list of edge attributes corresponding to the edges in the graph.
            y (List): A list of target values or labels associated with the graph data.
            num_nodes (List): A list containing the number of nodes in each graph.
            node_feat (List): A list of node features for each graph in the dataset.
            batch_info (Dict): A dictionary containing batch information for the graphs.

        Returns:
            None.

        Raises:
            TypeError: If the input parameters are not of the expected types.
            ValueError: If the input parameters do not meet specific requirements within the method logic.
            IndexError: If there are issues with index access during the processing of graph data.
        """
        features = []
        num_features = len(edge_index)
        for i in range(num_features):
            features.append({"edge_index": edge_index[i],
                             "edge_attr": edge_attr[i],
                             "y": y[i],
                             "num_nodes": num_nodes[i],
                             "node_feat": node_feat[i]})

        if self.on_the_fly_processing:
            features = [preprocess_item(i) for i in features]

        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        batch = {}

        max_node_num = max(len(i["input_nodes"]) for i in features)
        node_feat_size = len(features[0]["input_nodes"][0])
        edge_feat_size = len(features[0]["attn_edge_type"][0][0])
        max_dist = max(len(i["input_edges"][0][0]) for i in features)
        edge_input_size = len(features[0]["input_edges"][0][0][0])
        batch_size = len(features)

        batch["attn_bias"] = np.zeros((batch_size, max_node_num + 1, max_node_num + 1),
                                       dtype=np.float32)
        batch["attn_edge_type"] = np.zeros((batch_size, max_node_num, max_node_num, edge_feat_size),
                                            dtype=np.int64)
        batch["spatial_pos"] = np.zeros((batch_size, max_node_num, max_node_num),
                                         dtype=np.int64)
        batch["in_degree"] = np.zeros((batch_size, max_node_num),
                                       dtype=np.int64)
        batch["input_nodes"] = np.zeros((batch_size, max_node_num, node_feat_size),
                                         dtype=np.int64)
        batch["input_edges"] = np.zeros(
            (batch_size, max_node_num, max_node_num, max_dist, edge_input_size),
            dtype=np.int64
        )

        for idx, ftr in enumerate(features):

            if len(ftr["attn_bias"][1:, 1:][ftr["spatial_pos"] >= self.spatial_pos_max]) > 0:
                ftr["attn_bias"][1:, 1:][ftr["spatial_pos"] >= self.spatial_pos_max] = float("-inf")

            batch["attn_bias"][idx, : ftr["attn_bias"].shape[0], : ftr["attn_bias"].shape[1]] = ftr["attn_bias"]
            batch["attn_edge_type"][idx, : ftr["attn_edge_type"].shape[0], : ftr["attn_edge_type"].shape[1], :] = ftr[
                "attn_edge_type"
            ]
            batch["spatial_pos"][idx, : ftr["spatial_pos"].shape[0], : ftr["spatial_pos"].shape[1]] = ftr["spatial_pos"]
            batch["in_degree"][idx, : ftr["in_degree"].shape[0]] = ftr["in_degree"]
            batch["input_nodes"][idx, : ftr["input_nodes"].shape[0], :] = ftr["input_nodes"]
            batch["input_edges"][
                idx, : ftr["input_edges"].shape[0], : ftr["input_edges"].shape[1], : ftr["input_edges"].shape[2], :
            ] = ftr["input_edges"]

        batch["out_degree"] = batch["in_degree"]

        sample = features[0]["labels"]
        if len(sample) == 1:  # one task
            if isinstance(sample[0], float):  # regression
                batch["labels"] = np.concatenate([i["labels"] for i in features])
            else:  # binary classification
                batch["labels"] = np.concatenate([i["labels"] for i in features])
        else:  # multi task classification, left to float to keep the NaNs
            batch["labels"] = np.stack([i["labels"] for i in features], axis=0)

        outputs = [batch[key] for key in self.output_columns]
        return tuple(outputs)
