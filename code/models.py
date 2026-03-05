# src/models.py
import os
import pickle
from typing import Dict, Any, List
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm


class RidgeReconstruction:
    """
    一个封装了岭回归重建流程的类。
    - 使用 scikit-learn Pipeline 封装标准化和岭回归。
    - 使用 GridSearchCV 和 LeaveOneGroupOut (LOMO) 高效地进行交叉验证，寻找最优lambda。
    - 为每个独特的观测网络（mask_hash）训练一个专属的Pipeline。
    """

    def __init__(self, lambda_candidates: List[float]):
        """
        Args:
            lambda_candidates: 用于交叉验证的正则化强度alpha的候选列表。
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
        为每个独特的掩码分组训练模型。

        返回:
            model_library (Dict[Any, Pipeline]): 一个字典，键是mask_hash，值是训练好的scikit-learn Pipeline对象。
        """
        model_library: Dict[Any, Pipeline] = {}

        # 使用 tqdm 创建一个可视化的训练进度条
        progress_bar = tqdm(X_grouped.items(), desc="Training Models for each Mask")

        for mask_hash, X_group in progress_bar:
            Y_group = Y_grouped[mask_hash]
            groups = groups_grouped[mask_hash]
            n_samples, n_features = X_group.shape
            n_groups = len(np.unique(groups))

            progress_bar.set_postfix({
                "hash": f"{mask_hash % 10000:04d}",  # 显示哈希值的后4位
                "n_feat": n_features,
                "n_samp": n_samples,
                "n_groups": n_groups
            })

            # LOMO交叉验证至少需要2个组
            if n_groups < 2:
                print(f"\n[WARN] Skipping mask {mask_hash}: Not enough groups ({n_groups}) for CV.")
                continue

            # 1. 创建Pipeline：先标准化，再设置阈值，最后进行岭回归
            pipeline = make_pipeline(StandardScaler(),  VarianceThreshold(),Ridge())

            # 2. 定义参数网格：我们要优化的参数是岭回归的alpha (即lambda)
            # 'ridge__alpha' 是scikit-learn pipeline中命名参数的标准方式
            param_grid = {'ridge__alpha': self.lambda_candidates}

            # 3. 设置交叉验证
            logo = LeaveOneGroupOut()

            # 4. 初始化GridSearchCV
            # GridSearchCV 会自动处理：交叉验证、网格搜索、用最优参数在全部数据上重新训练
            grid_search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=logo,
                scoring='neg_mean_squared_error',
                n_jobs=-1,  # 使用所有CPU核心并行计算
                error_score='raise',
            )

            # 5. 执行训练
            try:
                grid_search.fit(X_group, Y_group, groups=groups)
                # self.model_library[mask_hash] = grid_search.best_estimator_
            except ValueError as e:
                print(f"\n[INFO] Skipping mask {mask_hash} due to training error: {e}")
                continue

            # 6. 保存最优模型
            # grid_search.best_estimator_ 是一个已经用最优lambda在全部数据上重新训练好的完整pipeline
            model_library[mask_hash] = grid_search.best_estimator_
            # self.model_library[mask_hash] = grid_search.best_estimator_

            # (可选) 打印最优lambda
            best_alpha = grid_search.best_params_['ridge__alpha']
            # print(f"  -> Best lambda for mask {mask_hash}: {best_alpha:.4f}")

        print(f"\n--- Model training finished. Successfully trained {len(model_library)} models. ---")
        return model_library


# --- 模型库保存/加载函数 ---
def save_model_library(model_library: Dict[Any, Pipeline], filepath: str) -> None:
    """用 pickle 保存模型库。"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(model_library, f)
    print(f"Model library with {len(model_library)} models saved to {filepath}")


def load_model_library(filepath: str) -> Dict[Any, Pipeline]:
    """从 pickle 文件中加载模型库。"""
    with open(filepath, "rb") as f:
        model_library = pickle.load(f)
    print(f"Model library with {len(model_library)} models loaded from {filepath}")
    return model_library