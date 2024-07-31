import math
import pickle
import random
from typing import List, Tuple

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from tqdm.auto import tqdm


class Solution:
    def __init__(self, n_estimators: int = 100, lr: float = 0.5, ndcg_top_k: int = 10,
                 subsample: float = 0.6, colsample_bytree: float = 0.9,
                 max_depth: int = 5, min_samples_leaf: int = 8):
        self.best_ndcg = None
        self._prepare_data()
        self.ndcg_top_k = ndcg_top_k
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.trees = []
        self.feature_indices = []

    def _get_data(self) -> List[np.ndarray]:
        train_df, test_df = msrank_10k()

        X_train = train_df.drop([0, 1], axis=1).values
        y_train = train_df[0].values
        query_ids_train = train_df[1].values.astype(int)

        X_test = test_df.drop([0, 1], axis=1).values
        y_test = test_df[0].values
        query_ids_test = test_df[1].values.astype(int)

        return [X_train, y_train, query_ids_train, X_test, y_test, query_ids_test]

    def _prepare_data(self) -> None:
        (X_train, y_train, self.query_ids_train,
         X_test, y_test, self.query_ids_test) = self._get_data()

        self.X_train = self._scale_features_in_query_groups(X_train, self.query_ids_train)
        self.X_test = self._scale_features_in_query_groups(X_test, self.query_ids_test)

        self.ys_train = torch.FloatTensor(np.nan_to_num(y_train, nan=0.0)).reshape(-1, 1)
        self.ys_test = torch.FloatTensor(np.nan_to_num(y_test, nan=0.0)).reshape(-1, 1)
        self.X_train = torch.FloatTensor(self.X_train)
        self.X_test = torch.FloatTensor(self.X_test)
        print('ok')

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        scaled_features = np.zeros_like(inp_feat_array)
        unique_query_ids = np.unique(inp_query_ids)

        for qid in unique_query_ids:
            mask = inp_query_ids == qid
            scaler = StandardScaler()
            scaled_features[mask] = scaler.fit_transform(inp_feat_array[mask])

        return scaled_features

    def _compute_lambdas(self, y_true, y_pred) -> torch.FloatTensor:
        ideal_dcg = self._compute_ideal_dcg(y_true)
        N = 1 / ideal_dcg

        _, rank_order = torch.sort(y_true, descending=True)
        rank_order += 1

        with torch.no_grad():
            pos_pairs_score_diff = 1.0 + torch.exp((y_pred - y_pred.t()))

            Sij = self._compute_labels_in_batch(y_true)
            gain_diff = self._compute_gain_diff(y_true)

            decay_diff = (1.0 / torch.log2(rank_order + 1.0)) - (1.0 / torch.log2(rank_order.t() + 1.0))
            delta_ndcg = torch.abs(N * gain_diff * decay_diff)
            lambda_update = (0.5 * (1 - Sij) - 1 / pos_pairs_score_diff) * delta_ndcg
            lambda_update = torch.sum(lambda_update, dim=1, keepdim=True)

        return lambda_update

    def _compute_labels_in_batch(self, y_true):
        rel_diff = y_true - y_true.t()
        pos_pairs = (rel_diff > 0).type(torch.float32)
        neg_pairs = (rel_diff < 0).type(torch.float32)
        Sij = pos_pairs - neg_pairs
        return Sij

    def _compute_gain_diff(self, y_true, gain_scheme='exp2'):
        if gain_scheme == "exp2":
            gain_diff = torch.pow(2.0, y_true) - torch.pow(2.0, y_true.t())
        elif gain_scheme == "diff":
            gain_diff = y_true - y_true.t()
        else:
            raise ValueError(f"{gain_scheme} method not supported")
        return gain_diff

    def _compute_ideal_dcg(self, y_true, ndcg_scheme='exp2'):
        sorted_labels, _ = torch.sort(y_true, descending=True)
        if ndcg_scheme == 'exp2':
            gains = torch.pow(2.0, sorted_labels) - 1
        else:
            gains = sorted_labels
        discounts = torch.log2(torch.arange(2, sorted_labels.size(0) + 2).float())
        return torch.sum(gains / discounts)

    def _train_one_tree(self, cur_tree_idx: int, train_preds: torch.FloatTensor) -> Tuple[
        DecisionTreeRegressor, np.ndarray]:
        lambdas = torch.zeros_like(self.ys_train)
        unique_queries = np.unique(self.query_ids_train)

        for qid in unique_queries:
            mask = self.query_ids_train == qid
            y_true_group = self.ys_train[mask]
            y_pred_group = train_preds[mask]
            lambdas[mask] = self._compute_lambdas(y_true_group, y_pred_group)

        np.random.seed(cur_tree_idx)
        sampled_indices = np.random.choice(len(self.X_train), int(self.subsample * len(self.X_train)), replace=False)
        sampled_features = np.random.choice(self.X_train.shape[1], int(self.colsample_bytree * self.X_train.shape[1]),
                                            replace=False)

        tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                     random_state=cur_tree_idx)
        self.lbd = lambdas
        self.si = sampled_indices
        self.sf = sampled_features
        tree.fit(self.X_train[sampled_indices][:, sampled_features],
                 np.nan_to_num(-lambdas[sampled_indices].numpy(), 0.0).flatten())

        return tree, sampled_features

    def _calc_data_ndcg(self, queries_list: np.ndarray, true_labels: torch.FloatTensor,
                        preds: torch.FloatTensor) -> float:
        unique_queries = np.unique(queries_list)
        ndcg_scores = []

        for qid in unique_queries:
            mask = queries_list == qid
            ndcg_score = self._ndcg_k(true_labels[mask], preds[mask], self.ndcg_top_k)
            ndcg_scores.append(ndcg_score)

        return np.mean(ndcg_scores)

    def fit(self):
        np.random.seed(0)
        train_preds = torch.zeros_like(self.ys_train)
        best_ndcg = -float('inf')
        best_trees = []

        for cur_tree_idx in tqdm(range(self.n_estimators)):
            tree, sampled_features = self._train_one_tree(cur_tree_idx, train_preds)
            self.trees.append(tree)
            self.feature_indices.append(sampled_features)

            new_preds = tree.predict(self.X_train[:, sampled_features])
            train_preds += self.lr * torch.FloatTensor(new_preds).reshape(-1, 1)

            valid_preds = self.predict(self.X_test)
            ndcg_score = self._calc_data_ndcg(self.query_ids_test, self.ys_test, valid_preds)

            if ndcg_score >= best_ndcg:
                best_ndcg = ndcg_score
                best_trees = self.trees[:]
                best_feature_indices = self.feature_indices[:]
            else:
                print(cur_tree_idx)
                # pr = train_preds
                print(train_preds)

            self.trees = best_trees
            self.feature_indices = best_feature_indices
            self.best_ndcg = best_ndcg

    def predict(self, data: torch.FloatTensor) -> torch.FloatTensor:
        preds = torch.zeros((data.size(0), 1))

        for tree, features in zip(self.trees, self.feature_indices):
            tree_preds = tree.predict(data[:, features])
            preds += self.lr * torch.FloatTensor(tree_preds).reshape(-1, 1)

        return preds

    def _ndcg_k(self, ys_true, ys_pred, ndcg_top_k) -> float:
        _, indices = torch.sort(ys_pred, descending=True)
        ideal_indices = torch.sort(ys_true, descending=True).indices
        dcg = 0.0
        idcg = 0.0

        k = min(ndcg_top_k, len(ys_true))

        for i in range(k):
            dcg += (2 ** ys_true[indices[i]] - 1) / math.log2(i + 2)
            idcg += (2 ** ys_true[ideal_indices[i]] - 1) / math.log2(i + 2)

        return dcg / idcg if idcg > 0 else 0.0

    def save_model(self, path: str):
        state = {
            'trees': self.trees,
            'feature_indices': self.feature_indices,
            'n_estimators': self.n_estimators,
            'lr': self.lr,
            'ndcg_top_k': self.ndcg_top_k,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_model(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.trees = state['trees']
        self.feature_indices = state['feature_indices']
        self.n_estimators = state['n_estimators']
        self.lr = state['lr']
        self.ndcg_top_k = state['ndcg_top_k']
        self.subsample = state['subsample']
        self.colsample_bytree = state['colsample_bytree']
        self.max_depth = state['max_depth']
        self.min_samples_leaf = state['min_samples_leaf']