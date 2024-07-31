import math

import numpy as np
import torch
from catboost.datasets import msrank_10k
from sklearn.preprocessing import StandardScaler

from typing import List


class ListNet(torch.nn.Module):
    def __init__(self, num_input_features: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model = torch.nn.Sequential(
            torch.nn.Linear(num_input_features, self.hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, input_1: torch.Tensor) -> torch.Tensor:
        logits = self.model(input_1)
        return logits


class Solution:
    def __init__(self, n_epochs: int = 5, listnet_hidden_dim: int = 30,
                 lr: float = 0.001, ndcg_top_k: int = 10):
        self._prepare_data()
        self.num_input_features = self.X_train.shape[1]
        self.ndcg_top_k = ndcg_top_k
        self.n_epochs = n_epochs
        self.model = self._create_model(
            self.num_input_features, listnet_hidden_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

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

        self.ys_train = torch.FloatTensor(y_train)
        self.ys_test = torch.FloatTensor(y_test)
        self.X_train = torch.FloatTensor(self.X_train)
        self.X_test = torch.FloatTensor(self.X_test)

    def _scale_features_in_query_groups(self, inp_feat_array: np.ndarray,
                                        inp_query_ids: np.ndarray) -> np.ndarray:
        scaled_features = np.zeros_like(inp_feat_array)
        unique_query_ids = np.unique(inp_query_ids)

        for qid in unique_query_ids:
            mask = inp_query_ids == qid
            scaler = StandardScaler()
            scaled_features[mask] = scaler.fit_transform(inp_feat_array[mask])

        return scaled_features

    def _create_model(self, listnet_num_input_features: int,
                      listnet_hidden_dim: int) -> torch.nn.Module:
        torch.manual_seed(0)
        net = ListNet(listnet_num_input_features, listnet_hidden_dim)
        return net

    def fit(self) -> List[float]:
        ndcg_scores = []

        for epoch in range(self.n_epochs):
            self._train_one_epoch()
            ndcg = self._eval_test_set()
            ndcg_scores.append(ndcg)

        return ndcg_scores

    def _calc_loss(self, batch_ys: torch.FloatTensor,
                   batch_pred: torch.FloatTensor) -> torch.FloatTensor:
        pred_probs = torch.nn.functional.softmax(batch_pred, dim=0)
        true_probs = torch.nn.functional.softmax(batch_ys, dim=0)
        loss = torch.nn.functional.kl_div(pred_probs.log(), true_probs, reduction='batchmean')
        return loss

    def _train_one_epoch(self) -> None:
        self.model.train()
        unique_query_ids = np.unique(self.query_ids_train)

        for qid in unique_query_ids:
            mask = self.query_ids_train == qid
            X_batch = self.X_train[mask]
            y_batch = self.ys_train[mask]

            self.optimizer.zero_grad()
            y_pred = self.model(X_batch).squeeze()
            loss = self._calc_loss(y_batch, y_pred)
            loss.backward()
            self.optimizer.step()

    def _eval_test_set(self) -> float:
        with torch.no_grad():
            self.model.eval()
            ndcgs = []
            unique_query_ids = np.unique(self.query_ids_test)

            for qid in unique_query_ids:
                mask = self.query_ids_test == qid
                X_batch = self.X_test[mask]
                y_batch = self.ys_test[mask]

                y_pred = self.model(X_batch).squeeze()
                ndcg = self._ndcg_k(y_batch, y_pred, self.ndcg_top_k)
                ndcgs.append(ndcg)

            return np.mean(ndcgs)

    def _ndcg_k(self, ys_true: torch.Tensor, ys_pred: torch.Tensor,
                ndcg_top_k: int) -> float:
        _, indices = torch.sort(ys_pred, descending=True)
        ideal_indices = torch.sort(ys_true, descending=True).indices
        dcg = 0.0
        idcg = 0.0

        k = min(ndcg_top_k, len(ys_true))

        for i in range(k):
            try:
                dcg += (2 ** ys_true[indices[i]] - 1) / math.log2(i + 2)
                idcg += (2 ** ys_true[ideal_indices[i]] - 1) / math.log2(i + 2)
            except:
                print(i)
                print(indices)
                print(ys_true)

        return dcg / idcg if idcg > 0 else 0.0