"""Utility functions for priors."""

import os
from typing import Union

import h5py
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
# from ticl.priors import GPPrior, MLPPrior, ClassificationAdapterPrior, BooleanConjunctionPrior, StepFunctionPrior

from .config import get_ticl_prior_config


# def build_ticl_prior(prior_type: str, base_prior_type: str = None, max_num_classes: int = None) -> Union[MLPPrior, GPPrior, ClassificationAdapterPrior, BooleanConjunctionPrior, StepFunctionPrior]:
#     """Builds a TICL prior based on the prior type string using the defaults in config.py."""

#     cfg = get_ticl_prior_config(prior_type, max_num_classes)
    
#     if prior_type == "mlp":
#         return MLPPrior(cfg)
#     elif prior_type == "gp":
#         return GPPrior(cfg)
#     elif prior_type == "classification_adapter":
#         if base_prior_type is None:
#             base_prior_type = "mlp"  # default to MLP
#         # build the base regression prior
#         base_prior = build_ticl_prior(base_prior_type)
#         return ClassificationAdapterPrior(base_prior, **cfg)
#     elif prior_type == "boolean_conjunctions":
#         return BooleanConjunctionPrior(hyperparameters=cfg)
#     elif prior_type == "step_function":
#         return StepFunctionPrior(cfg)
#     else:
#         raise ValueError(f"Unsupported TICL prior type: {prior_type}")


def dump_prior_to_h5(
    prior, 
    max_classes: int, 
    batch_size: int, 
    save_path: str, 
    problem_type: str, 
    max_seq_len: int,
    max_features: int,
    save_graph_info: bool = False,
    save_diagnostics: bool = False,
    resume: bool = False,
):
    """Dumps synthetic prior data into an HDF5 file for later training.

    If ``save_diagnostics`` is True (gcfm_tabicl path), the target-selection
    diagnostics are persisted alongside X/y: per-node marginal variance and
    topological depth (ragged, full graph length), the selected target's id /
    depth / variance, the rule's eligible-pool size, the per-dataset rejection
    count, and the full-graph adjacency (flattened, ragged) for varsortability.
    """

    with h5py.File(save_path, "w") as f:
        dump_X = f.create_dataset(
            "X",
            shape=(0, max_seq_len, max_features),
            maxshape=(None, max_seq_len, max_features),
            chunks=(batch_size, max_seq_len, max_features),
            compression="lzf",
        )
        dump_num_features = f.create_dataset(
            "num_features", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
        )
        dump_num_datapoints = f.create_dataset(
            "num_datapoints", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
        )
        dump_y = f.create_dataset(
            "y", shape=(0, max_seq_len), maxshape=(None, max_seq_len), chunks=(batch_size, max_seq_len)
        )
        dump_single_eval_pos = f.create_dataset(
            "single_eval_pos", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
        )

        if save_graph_info:
            max_nodes = max_features + 1  # +1 for the target node
            dump_adj = f.create_dataset(
                "adj",
                shape=(0, max_nodes, max_nodes),
                maxshape=(None, max_nodes, max_nodes),
                chunks=(batch_size, max_nodes, max_nodes),
            )
            dump_density = f.create_dataset(
                "density", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="f4"
            )

        if save_diagnostics:
            vlen_f4 = h5py.special_dtype(vlen=np.dtype("f4"))
            vlen_i4 = h5py.special_dtype(vlen=np.dtype("i4"))
            # Ragged, full-graph-length per-node arrays.
            dump_node_variances = f.create_dataset(
                "node_variances", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype=vlen_f4
            )
            dump_node_depths = f.create_dataset(
                "node_depths", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype=vlen_i4
            )
            # Full-graph adjacency, flattened row-major (reshape with num_nodes on read).
            dump_adj_full = f.create_dataset(
                "adj_full_flat", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype=vlen_f4
            )
            # Per-dataset scalars.
            dump_num_nodes = f.create_dataset(
                "num_nodes", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
            )
            dump_target_node = f.create_dataset(
                "target_node", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
            )
            dump_target_depth = f.create_dataset(
                "target_depth", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
            )
            dump_target_variance = f.create_dataset(
                "target_variance", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="f4"
            )
            dump_eligible_pool_size = f.create_dataset(
                "eligible_pool_size", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
            )
            dump_n_rejections = f.create_dataset(
                "n_rejections", shape=(0,), maxshape=(None,), chunks=(batch_size,), dtype="i4"
            )

        if problem_type == "classification" and max_classes is not None:
            f.create_dataset("max_num_classes", data=np.array((max_classes,)), chunks=(1,))
        f.create_dataset("original_batch_size", data=np.array((batch_size,)), chunks=(1,))
        f.create_dataset("problem_type", data=problem_type, dtype=h5py.string_dtype())

        for batch_idx, e in tqdm(enumerate(prior), total=len(prior)):
            x = e["x"].to("cpu").numpy()
            y = e["y"].to("cpu").numpy()
            single_eval_pos = e["single_eval_pos"]
            if isinstance(single_eval_pos, torch.Tensor):
                single_eval_pos = single_eval_pos.item()

            if save_graph_info:
                adj = e['adj'].to('cpu').numpy() 
                density = e['density'].to('cpu').numpy()

            # pad x and y to the maximum sequence length and number of features needed for tabicl
            x_padded = np.pad(
                x, ((0, 0), (0, max_seq_len - x.shape[1]), (0, max_features - x.shape[2])), mode="constant"
            )
            y_padded = np.pad(y, ((0, 0), (0, max_seq_len - y.shape[1])), mode="constant")

            dump_X.resize(dump_X.shape[0] + batch_size, axis=0)
            dump_X[-batch_size:] = x_padded

            dump_y.resize(dump_y.shape[0] + batch_size, axis=0)
            dump_y[-batch_size:] = y_padded

            dump_num_features.resize(dump_num_features.shape[0] + batch_size, axis=0)
            dump_num_features[-batch_size:] = x.shape[2]

            dump_num_datapoints.resize(dump_num_datapoints.shape[0] + batch_size, axis=0)
            dump_num_datapoints[-batch_size:] = x.shape[1]

            dump_single_eval_pos.resize(dump_single_eval_pos.shape[0] + batch_size, axis=0)
            dump_single_eval_pos[-batch_size:] = single_eval_pos

            if save_graph_info:
                dump_adj.resize(dump_adj.shape[0] + batch_size, axis=0)
                dump_adj[-batch_size:] = adj

                dump_density.resize(dump_density.shape[0] + batch_size, axis=0)
                dump_density[-batch_size:] = density

            if save_diagnostics:
                node_variances = e["node_variances"]      # list[batch] of per-node lists
                node_depths = e["node_depths"]
                adj_full = e["adj_full"].to("cpu").numpy()  # (batch, n, n) — n constant within a batch

                def _resize_append(ds, values):
                    ds.resize(ds.shape[0] + batch_size, axis=0)
                    ds[-batch_size:] = values

                _resize_append(dump_node_variances, [np.asarray(v, dtype="f4") for v in node_variances])
                _resize_append(dump_node_depths, [np.asarray(d, dtype="i4") for d in node_depths])
                _resize_append(dump_adj_full, [adj_full[i].astype("f4").ravel() for i in range(batch_size)])
                _resize_append(dump_num_nodes, [len(v) for v in node_variances])
                _resize_append(dump_target_node, e["target_node"])
                _resize_append(dump_target_depth, e["target_depth"])
                _resize_append(dump_target_variance, e["target_variance"])
                _resize_append(dump_eligible_pool_size, e["eligible_pool_size"])
                _resize_append(dump_n_rejections, e["n_rejections"])

            # Periodic flush for crash safety
            if (batch_idx + 1) % 50 == 0:
                f.flush()