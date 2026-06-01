"""
models.py — Ridge regression reconstruction model.

Provides:
  - RidgeReconstruction: trains one Ridge model per unique observation
    pattern (mask hash) using Leave-One-Group-Out cross-validation.
  - save_model_library / load_model_library: persist/restore trained models.
"""

import os
import pickle
from typing import Dict, Any, List

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import make_pipeline, Pipeline
from tqdm import tqdm


class RidgeReconstruction:
    """
    Ridge regression reconstruction with LOMO cross-validation.

    For each unique observation mask (identified by a hash of its binary
    pattern), a separate scikit-learn Pipeline is trained:
        StandardScaler → VarianceThreshold → Ridge

    The regularization strength λ is selected via GridSearchCV over
    a candidate list using Leave-One-Group-Out (LOMO) cross-validation,
    where each group corresponds to one CMIP6 model.
    """

    def __init__(self, lambda_candidates: List[float]):
        """
        Args:
            lambda_candidates: Candidate α (regularization strength) values
                              for grid search.
        """
        self.lambda_candidates = lambda_candidates

    def train(
        self,
        X_grouped: Dict[Any, np.ndarray],
        Y_grouped: Dict[Any, np.ndarray],
        groups_grouped: Dict[Any, np.ndarray],
        hash_to_mask_map: Dict[Any, np.ndarray],
        output_dir: str,
    ) -> Dict[Any, Pipeline]:
        """
        Train one model per unique mask pattern.

        Args:
            X_grouped: {mask_hash: feature matrix (n_samples, n_features)}
            Y_grouped: {mask_hash: target vector (n_samples,)}
            groups_grouped: {mask_hash: group labels (n_samples,)}
            hash_to_mask_map: {mask_hash: 2D boolean mask}
            output_dir: Directory for saving intermediate results.

        Returns:
            model_library: {mask_hash: trained Pipeline}
        """
        model_library: Dict[Any, Pipeline] = {}

        progress_bar = tqdm(
            X_grouped.items(),
            desc="Training models per mask"
        )

        for mask_hash, X_group in progress_bar:
            Y_group = Y_grouped[mask_hash]
            groups = groups_grouped[mask_hash]
            n_samples, n_features = X_group.shape
            n_groups = len(np.unique(groups))

            progress_bar.set_postfix({
                "hash": f"{mask_hash % 10000:04d}",
                "n_feat": n_features,
                "n_samp": n_samples,
                "n_groups": n_groups,
            })

            # LOMO requires at least 2 groups
            if n_groups < 2:
                print(f"\n[WARN] Skipping mask {mask_hash}: "
                      f"only {n_groups} group(s).")
                continue

            # Build pipeline
            pipeline = make_pipeline(
                StandardScaler(),
                VarianceThreshold(),
                Ridge(),
            )

            param_grid = {'ridge__alpha': self.lambda_candidates}
            logo = LeaveOneGroupOut()

            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=logo,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                error_score='raise',
            )

            try:
                grid_search.fit(X_group, Y_group, groups=groups)
            except ValueError as e:
                print(f"\n[INFO] Skipping mask {mask_hash}: {e}")
                continue

            model_library[mask_hash] = grid_search.best_estimator_

        print(f"\n--- Training finished. "
              f"{len(model_library)} model(s) trained. ---")
        return model_library


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def save_model_library(
    model_library: Dict[Any, Pipeline],
    filepath: str,
) -> None:
    """Persist model library to disk via pickle."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model_library, f)
    print(f"Model library ({len(model_library)} models) saved to "
          f"{filepath}")


def load_model_library(filepath: str) -> Dict[Any, Pipeline]:
    """Load model library from pickle file."""
    with open(filepath, "rb") as f:
        model_library = pickle.load(f)
    print(f"Model library ({len(model_library)} models) loaded from "
          f"{filepath}")
    return model_library
