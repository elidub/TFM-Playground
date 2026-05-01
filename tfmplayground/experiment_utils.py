from typing import TYPE_CHECKING, Tuple, Dict, List
from matplotlib import pyplot as plt
import functools
from sklearn.linear_model import LinearRegression, LogisticRegression
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score, root_mean_squared_error
import numpy as np
from sklearn.model_selection import StratifiedKFold
import seaborn as sns
import numpy as np
import openml
import pandas as pd
from tqdm import tqdm
from openml.tasks import TaskType
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
"""
=================== DATA LOADING AND PREPROCESSING ===================
"""

def get_feature_preprocessor(X: np.ndarray | pd.DataFrame) -> ColumnTransformer:
    """
    fits a preprocessor that imputes NaNs, encodes categorical features and removes constant features
    """
    X = pd.DataFrame(X)
    num_mask = []
    cat_mask = []
    for col in X:
        unique_non_nan_entries = X[col].dropna().unique()
        if len(unique_non_nan_entries) <= 1:
            num_mask.append(False)
            cat_mask.append(False)
            continue
        non_nan_entries = X[col].notna().sum()
        numeric_entries = pd.to_numeric(X[col], errors='coerce').notna().sum() # in case numeric columns are stored as strings
        num_mask.append(non_nan_entries == numeric_entries)
        cat_mask.append(non_nan_entries != numeric_entries)
        # num_mask.append(is_numeric_dtype(X[col]))  # Assumes pandas dtype is correct

    num_mask = np.array(num_mask)
    cat_mask = np.array(cat_mask)

    num_transformer = Pipeline([
        ("to_pandas", FunctionTransformer(lambda x: pd.DataFrame(x) if not isinstance(x, pd.DataFrame) else x)), # to apply pd.to_numeric of pandas
        ("to_numeric", FunctionTransformer(lambda x: x.apply(pd.to_numeric, errors='coerce').to_numpy())), # in case numeric columns are stored as strings
        ("imputer", SimpleImputer(strategy="mean")),
    ])
    cat_transformer = Pipeline([
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)),
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_mask),
            ('cat', cat_transformer, cat_mask)
        ]
    )
    return preprocessor

def get_openml_datasets(
        max_features_eval: int | None,
        new_instances_eval: int | None,
        target_classes_filter: int | None,
        eval_subsample_features: List[int] | int | None,
        eval_subsample_samples: int | None,
        tabarena_light: bool,
        seed: int = 0,
        test_fraction: float = 1/3, # following TabArena's ~0.33 test fraction across datasets
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load OpenML tabarena datasets with optional feature and row subsampling.

    Parameters
    ----------
    max_features_eval : int | None
        Maximum number of features a dataset may have to be included. None = no filter.
    new_instances_eval : int | None
        Maximum number of instances to keep via stratified subsampling. None = no subsampling.
    target_classes_filter : int | None
        Maximum number of target classes (0 = regression). None = no filter.
    eval_subsample_features : List[int] | int | None
        If a single int and the dataset has more features than this value, randomly
        subsample down to this many features (seeded).
        If a list of ints, the dataset is subsampled for each value in the list and
        stored under the key f"{dataset.name}_nfeat{n}" for each n. Datasets with
        fewer features than a given n are skipped for that n.
    eval_subsample_samples : int | None
        If set, the test split is subsampled to at most this many rows. The train
        split is subsampled proportionally using the fixed ~0.33 test fraction
        (i.e. train cap = eval_subsample_samples * 2), both via stratified subsampling.
    tabarena_light : bool
        If True, use 1 repeat and 1 fold (fast). If False, use 10 repeats and 3 folds
        (full TabArena evaluation protocol). Keys include a repeat/fold suffix only
        when tabarena_light=False.
    seed : int
        Global random seed used for all stochastic operations.

    Returns
    -------
    dict mapping dataset name -> (X_train, y_train, X_test, y_test) as numpy arrays.
    When tabarena_light=False, keys are "{dataset_name}_repeat{r}_fold{f}{feature_suffix}".
    When tabarena_light=True, keys are "{dataset_name}{feature_suffix}".
    """
    task_ids = [
        363612, 363613, 363614, 363615, 363616, 363618, 363619, 363620,
        363621, 363623, 363624, 363625, 363626, 363627, 363628, 363629,
        363630, 363631, 363632, 363671, 363672, 363673, 363674, 363675,
        363676, 363677, 363678, 363679, 363681, 363682, 363683, 363684,
        363685, 363686, 363689, 363691, 363693, 363694, 363696, 363697,
        363698, 363699, 363700, 363702, 363704, 363705, 363706, 363707,
        363708, 363711, 363712,
    ]  # TabArena v0.1

    n_repeats, n_folds = (1, 1) if tabarena_light else (10, 3)

    classification: bool = target_classes_filter is None or target_classes_filter > 0

    # Normalise eval_subsample_features into a list of (n_features, key_suffix) pairs.
    # - None          → [(None, "")]          one pass, no suffix, no subsampling
    # - int           → [(n,    "")]          one pass, no suffix, subsample if needed
    # - List[int]     → [(n, f"_nfeat{n}")]  one pass per value, always suffixed
    if isinstance(eval_subsample_features, list):
        subsample_plan: list[tuple[int | None, str]] = [
            (n, f"-nfeat{n}") for n in eval_subsample_features
        ]
    else:
        subsample_plan = [(eval_subsample_features, "")]

    def _subsample_split(
        X: pd.DataFrame,
        y: pd.Series,
        max_samples: int,
        split_seed: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Stratified row subsampling, applied only when needed."""
        if max_samples >= len(y):
            return X, y
        y_stratify = y if classification else pd.qcut(y, q=5, labels=False, duplicates="drop")
        _, X_sub, _, y_sub = train_test_split(
            X, y,
            test_size=max_samples,
            stratify=y_stratify,
            random_state=split_seed,
        )
        return X_sub.reset_index(drop=True), y_sub.reset_index(drop=True)

    datasets = {}
    for task_counter, task_id in enumerate(task_ids):
        task = openml.tasks.get_task(task_id, download_splits=False)

        # ── task-type filter ────────────────────────────────────────────────
        if classification and task.task_type_id != TaskType.SUPERVISED_CLASSIFICATION:
            continue
        if not classification and task.task_type_id != TaskType.SUPERVISED_REGRESSION:
            continue

        dataset = task.get_dataset(download_data=False)

        # ── quality filter ──────────────────────────────────────────────────
        q = dataset.qualities
        if (
            (max_features_eval is not None and q["NumberOfFeatures"] > max_features_eval)
            or (new_instances_eval is not None and q["NumberOfInstances"] > new_instances_eval)
            or (target_classes_filter is not None and q["NumberOfClasses"] > target_classes_filter)
        ):
            continue

        # ── load full data once per dataset ─────────────────────────────────
        X_full, y_full, categorical_indicator, attribute_names = dataset.get_data(
            target=task.target_name, dataset_format="dataframe"
        )

        for repeat in range(n_repeats):
            for fold in range(n_folds):

                # Use a deterministic seed that varies per repeat/fold
                split_seed = seed + repeat * n_folds + fold

                # ── train/test split via OpenML indices ──────────────────────
                train_indices, test_indices = task.get_train_test_split_indices(
                    fold=fold, repeat=repeat
                )

                X_train = X_full.iloc[train_indices].reset_index(drop=True)
                y_train = y_full.iloc[train_indices].reset_index(drop=True)
                X_test  = X_full.iloc[test_indices].reset_index(drop=True)
                y_test  = y_full.iloc[test_indices].reset_index(drop=True)

                # ── row subsampling on both splits ───────────────────────────
                if eval_subsample_samples is not None:
                    test_max  = round(eval_subsample_samples * test_fraction)
                    train_max = eval_subsample_samples - test_max  # guarantees exact total
                    X_test,  y_test  = _subsample_split(X_test,  y_test,  test_max,  split_seed)
                    X_train, y_train = _subsample_split(X_train, y_train, train_max, split_seed)


                # ── encode y once (shared across all feature subsample variants)
                def _encode_y(y: pd.Series) -> np.ndarray:
                    y_np = y.to_numpy(copy=True)
                    if classification:
                        return LabelEncoder().fit_transform(y_np)
                    else:
                        return StandardScaler().fit_transform(y_np.reshape(-1, 1)).reshape(-1)

                y_train_np = _encode_y(y_train)
                y_test_np  = _encode_y(y_test)

                # ── feature subsampling plan ─────────────────────────────────
                len_features = X_train.shape[1]

                for n_features, key_suffix in subsample_plan:
                    if n_features is not None and len_features < n_features:
                        if key_suffix:  # list mode — skip silently
                            print(
                                f"skipping {dataset.name}{key_suffix}\n"
                                f"\t{len_features = } features < {n_features = } requested"
                            )
                            continue
                        # scalar mode — fall through without subsampling
                        X_train_sub = X_train.copy()
                        X_test_sub  = X_test.copy()
                    elif n_features is not None:
                        rng = np.random.default_rng(seed)
                        feature_choices = rng.choice(len_features, size=n_features, replace=False)
                        X_train_sub = X_train.iloc[:, feature_choices]
                        X_test_sub  = X_test.iloc[:, feature_choices]
                    else:
                        X_train_sub = X_train.copy()
                        X_test_sub  = X_test.copy()

                    # ── preprocessing & encoding ─────────────────────────────
                    # Fit preprocessor on train, apply to both splits
                    X_train_np = X_train_sub.to_numpy(copy=True)
                    X_test_np  = X_test_sub.to_numpy(copy=True)
                    preprocessor = get_feature_preprocessor(X_train_np)
                    X_train_np = preprocessor.fit_transform(X_train_np)
                    X_test_np  = preprocessor.transform(X_test_np)

                    # ── build key ────────────────────────────────────────────
                    # split_suffix = f"_repeat{repeat}_fold{fold}"
                    key = (f"{dataset.name}{key_suffix}", repeat, fold)
                    datasets[key] = (X_train_np, y_train_np, X_test_np, y_test_np)

    return datasets




"""
=================== EVALUATION ===================
"""


def _eval_model(model, datasets, classification: bool):
    """Evaluates a model on multiple datasets and returns metrics"""
    _skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    metrics = {}
    avg_metrics = {}
    for dataset_name, (X, y) in tqdm(datasets.items(), desc=f"Evaluating {model}", total=len(datasets), leave=False):
        targets = []
        probabilities = []
        
        y_stratify = y if classification else pd.qcut(y, q=5, labels=False, duplicates='drop')
        for train_idx, test_idx in _skf.split(X, y_stratify):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test  = y[train_idx], y[test_idx]
            targets.append(y_test)
            model.fit(X_train, y_train)

            if classification:
                y_proba = model.predict_proba(X_test)
                if y_proba.shape[1] == 2:  # binary classification with neural network
                    y_proba = y_proba[:, 1]
                probabilities.append(y_proba)
            else:
                y_pred = model.predict(X_test)
                probabilities.append(y_pred)
    
        targets = np.concatenate(targets, axis=0)
        probabilities = np.concatenate(probabilities, axis=0)

        if classification:
            metrics[f"{dataset_name}/roc_auc"] = roc_auc_score(targets, probabilities, multi_class="ovr")
        else:
            metrics[f"{dataset_name}/rmse"] = root_mean_squared_error(targets, probabilities)
    
    metric_names = list({key.split("/")[-1] for key in metrics.keys()})
    for metric_name in metric_names:
        avg_metric = np.mean([metrics[key] for key in metrics.keys() if key.endswith(metric_name)])
        avg_metrics[f"{metric_name}"] = float(avg_metric)
    
    return metrics, avg_metrics

def eval_model(model, datasets, classification: bool):
    """Evaluates a model on multiple datasets and returns metrics"""
    metrics = {}
    avg_metrics = {}
    for (dataset_name, repeat, fold), (X_train, y_train, X_test, y_test) in tqdm(datasets.items(), desc=f"Evaluating {model}", total=len(datasets), leave=False):
        model.fit(X_train, y_train)
        if classification:
            y_proba = model.predict_proba(X_test)
            if y_proba.shape[1] == 2:  # binary classification with neural network
                y_proba = y_proba[:, 1]
            else:
                raise NotImplementedError("Only binary classification with predict_proba output of shape (n_samples, 2) is currently supported.")
        else:
            y_pred = model.predict(X_test)

        if classification:
            score = roc_auc_score(y_test, y_proba, multi_class="ovr")
            metric = 'roc_auc'
        else:
            score = root_mean_squared_error(y_test, y_pred)
            metric = 'rmse'

        # metrics[f"{dataset_name}/repeat{repeat}_fold{fold}/{metric}"] = score
        metrics[(dataset_name, repeat, fold, metric)] = score

    df_score = pd.DataFrame(
        [(dataset, repeat, fold, metric, value) for (dataset, repeat, fold, metric), value in metrics.items()],
        columns=['dataset', 'repeat', 'fold', 'metric', 'metric_value']
    )

    return df_score

    metric_names = list({key.split("/")[-1] for key in metrics.keys()})
    for metric_name in metric_names:
        avg_metric = np.mean([metrics[key] for key in metrics.keys() if key.endswith(metric_name)])
        avg_metrics[f"{metric_name}"] = float(avg_metric)
    
    return metrics, avg_metrics

"""
=================== PLOTTING ===================
"""

def plot_runs(
        ax: plt.Axes, 
        runs: list[pd.DataFrame], 
        metric: str, 
        baselines: pd.DataFrame = None,
        baselines_std: pd.DataFrame = None,
        show_legend: bool = True,
        show_xlabel: bool = True,
        show_ylabel: bool = True,
        show_xtics: bool = True
        ):
    """
    Plots the run for a given metric and adds baselines

    `runs` is a list of dataframes where each dataframe corresponds 
    to a run from a model with the same config but a different seed.
    Each dataframe needs to have a `"training_time"` column and a `metric`column.

    `baselines` is a DataFrame with metric columns and rows whose index correspond to
    a ML algorithm.
    """
    colors = sns.color_palette("tab10")[1:]
    linestyles = [
        '--',      # dashed
        '-.',      # dash-dot
        ':',       # dotted
        (0, (3, 1, 1, 1)),  # dash-dot-dot
        (0, (5, 5))         # spaced dash
    ]

    training_times = [run["training_time"].tolist() for run in runs]
    training_times = sorted(set([item for sublist in training_times for item in sublist]))
    shared_time_runs = []
    for run in runs:
        run = run.copy()
        run = run[[metric, "training_time"]].set_index("training_time").reindex(training_times)
        run = run.interpolate()
        shared_time_runs.append(run)
    all_runs = pd.concat(shared_time_runs, axis=1).dropna()
    
    # plot mean and std of all runs or single run if only one run
    mean = all_runs.mean(axis=1)
    ax.plot(mean.index, mean, label="nanoTabPFN", zorder=2, color="blue")
    if all_runs.shape[1] > 1: # more than one run
        std = all_runs.std(axis=1)
        ax.fill_between(mean.index, mean - std, mean + std, alpha=0.2, zorder=2)

    # plot horizontal lines for baselines
    if baselines is not None:
        for i ,(baseline_name, baseline_value) in enumerate(baselines[metric].items()):
            # draw a horizontal line that ends at the same x as the runs
            color = colors[i % len(colors)]
            ax.plot([0, max(training_times)], [baseline_value, baseline_value], label=baseline_name, alpha=0.7, linestyle=linestyles[i], color=color, zorder=1)
            if baselines_std is not None and baseline_name in baselines_std.index:
                    std = baselines_std.loc[baseline_name, metric]
                    ax.fill_between([0, max(training_times)], [baseline_value - std, baseline_value - std], [baseline_value + std, baseline_value + std], alpha=0.2, zorder=1)
    
    # Plot Style
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.tick_params(axis="y", length=0)
    if not show_xtics:
        ax.tick_params(axis="x", length=0)
    if show_xlabel:
        ax.set_xlabel("Training time (seconds)")
    if show_ylabel:
        ax.set_ylabel(metric.split("/")[-1])
    max_time = max(training_times)
    ax.set_xlim(0, max_time)
    ylim = ax.get_ylim()
    ax.set_ylim(ylim[0], 1)
    
    # order legend entries by their y-value at the end of the plot
    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        label_y_values = {}
        for handle, label in zip(handles, labels):
            if isinstance(handle, plt.Line2D):
                y_data = handle.get_ydata()
                label_y_values[label] = y_data[-1]
        sorted_labels = sorted(label_y_values.items(), key=lambda x: x[1], reverse=True)
        sorted_handles = [handle for label, _ in sorted_labels for handle, lbl in zip(handles, labels) if lbl == label]
        sorted_labels = [label for label, _ in sorted_labels]
        ax.legend(sorted_handles, sorted_labels)
        
    # remove border
    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_run_grid(runs: list[pd.DataFrame], baselines: pd.DataFrame = None, baselines_std: pd.DataFrame = None, metric: str = "roc_auc") -> Tuple[plt.Figure, np.ndarray]:
    """Plots the runs in a grid of metrics x datasets"""
    # drop all columns without "/" for dataset/metric format
    datasets = list(set([col.split("/")[0] for col in runs[0].columns if "/" in col]))
    figsize = (len(datasets) * 4, 4.6)
    fig, axs = plt.subplots(1, len(datasets), figsize=figsize, sharex=True, sharey=True, layout="constrained")
    fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0.1)
    # Plot each metric and dataset
    for j, dataset in enumerate(datasets):
        ax = axs[j]
        plot_runs(ax, runs, f"{dataset}/{metric}", baselines, baselines_std, show_legend=False, show_xlabel=False, show_ylabel=(j==0))
        ax.set_title(dataset)
    fig.supxlabel("Training Time (seconds)")
    
    # y-axis and x-axis labels should have the same size as supxlabel
    for ax in axs.flatten():
        font_size = fig.texts[-1].get_fontsize() 
        ax.xaxis.label.set_size(font_size)
        ax.yaxis.label.set_size(font_size)
    
    # Create a single legend for the entire figure
    legend_handels_labels = [list(zip(*ax.get_legend_handles_labels())) for ax in axs.flatten()]
    legend_handels_labels = functools.reduce(lambda a, b: a + b, legend_handels_labels)
    unique = dict([(label, handle) for (handle, label) in legend_handels_labels])
    fig.legend(unique.values(), unique.keys(), loc="outside upper center", ncol=3)   
    return fig, axs

def get_baseline_results(
    # open_ml_datasets_kwargs: dict,
    datasets: dict,
    classification: bool,
    num_seeds: int = 1,  # If you want to reproduce the paper results, use 20 seeds
    include_tabpfn: bool = True,
) -> Tuple[Dict[str, Tuple[torch.Tensor, torch.Tensor]], pd.DataFrame, pd.DataFrame]:
    """
    Reproducing the training and evalaution of the NanoTabPFNPlayground paper notebook.
    """
    # datasets = get_openml_datasets(**open_ml_datasets_kwargs)

    # classification = open_ml_datasets_kwargs['target_classes_filter'] > 0

    if include_tabpfn:
        from tabpfn import TabPFNClassifier
        from tabpfn.config import ModelInterfaceConfig, PreprocessorConfig


        no_preprocessing_inference_config = ModelInterfaceConfig(
            FINGERPRINT_FEATURE=False,
            PREPROCESS_TRANSFORMS=[PreprocessorConfig(name='none')]
        )

    if classification:
        baseline_models = {
            # "TabPFN v2": [TabPFNClassifier(random_state=i) for i in range(num_seeds)],
            # "TabPFN v2 (no preprocessing)": [TabPFNClassifier(inference_config=no_preprocessing_inference_config, n_estimators=1, random_state=i) for i in range(num_seeds)],
            "Random Forest": [RandomForestClassifier(random_state=i) for i in range(num_seeds)],
            # "K-Nearest Neighbors": [KNeighborsClassifier()],
            "Decision Tree": [DecisionTreeClassifier(random_state=i) for i in range(num_seeds)],
            # "Linear" : [LogisticRegression(max_iter=1000, ) for i in range(num_seeds)],
        }
    else:
        baseline_models = {
            # "TabPFN v2": [TabPFNRegressor(random_state=i) for i in range(num_seeds)],
            # "Random Forest": [RandomForestRegressor(random_state=i) for i in range(num_seeds)],
            # "K-Nearest Neighbors": [KNeighborsRegressor()],
            "Decision Tree": [DecisionTreeRegressor(random_state=i) for i in range(num_seeds)],
            # "Linear" : [LinearRegression()],
        }

    df_scores = []
    for name, models in baseline_models.items():
        for seed, model in enumerate(models):
            df_score = eval_model(model, datasets=datasets, classification=classification)
            df_score['method'] = name
            df_score['seed'] = seed
            df_scores.append(df_score)
    df_scores = pd.concat(df_scores, ignore_index=True)
    return df_scores

    baseline_models_eval = {name: [eval_model(model, datasets=datasets, classification=classification) for model in models] for name, models in baseline_models.items()}

    def apply_aggregation(eval_results: dict, func=np.mean):
        aggregated_result = {}
        for result in eval_results:
            for metric, value in result.items():
                if metric not in aggregated_result:
                    aggregated_result[metric] = []
                aggregated_result[metric].append(value)
        for metric in aggregated_result:
            aggregated_result[metric] = func(aggregated_result[metric])
        return aggregated_result

    baselines = pd.DataFrame({
        name: apply_aggregation(models, np.mean) for name, models in baseline_models_eval.items()
    }).T

    baselines_std = pd.DataFrame({
        name: apply_aggregation(models, np.std) for name, models in baseline_models_eval.items()
    }).T

    return datasets, baselines, baselines_std