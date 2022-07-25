import imp

import torch
import torch.nn as nn


class VariableSizedEmbedding(nn.Module):
    
    def __init__(self, embedding_sizes: torch.Tensor, embedding_dimension: int, device: torch.device) -> None:
       
        super().__init__()
        self.device = device
        self.n_hidden = 64
        self.embedding_dimension = embedding_dimension

        values, self.indices = torch.sort(embedding_sizes)
        self.inverse_indices = torch.argsort(self.indices)

        unique_sizes, counts = torch.unique_consecutive(
            values, return_inverse=False, return_counts=True, dim=None)
        self.unique_sizes = unique_sizes

        self.anchor_embeddings = nn.ModuleList()
        for i in range(len(unique_sizes)):
            self.anchor_embeddings.append(nn.Embedding(int(counts[i]), embedding_dim=int(unique_sizes[i]), padding_idx=0).to(device))

        self.mlp = nn.ModuleList()

        for i in range(len(unique_sizes)):
            self.mlp.append(nn.Sequential(
                nn.Linear(int(unique_sizes[i]), self.n_hidden, device=self.device),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.embedding_dimension)
            ))

        cummulative_counts = torch.cumsum(counts, dim=0)
        self.start_offsets = torch.cat(
            (torch.Tensor([0]), cummulative_counts[:-1]))
        self.end_offsets = cummulative_counts

    def reset_parameters(self) -> None:
        for i in range(len(self.unique_sizes)):
            torch.nn.init.xavier_uniform_(self.anchor_embeddings[i].weight)
        for i in range(len(self.unique_sizes)):
            torch.nn.init.xavier_uniform_(self.mlp[i][0].weight)
            torch.nn.init.xavier_uniform_(self.mlp[i][2].weight)

    def forward(self, input: torch.Tensor):
        # Extra padding token replaced with index 0 (padding id) to not throw out of bounds error
        input[input == self.inverse_indices.shape[0]] = 0

        inversed_input_indices = self.inverse_indices[input]
        outputs = torch.zeros(input.shape[0], input.shape[1], self.embedding_dimension, device=self.device)

        for MLP_index, (start, end) in enumerate(zip(self.start_offsets, self.end_offsets)):
            mask = torch.logical_and(
                inversed_input_indices >= start,
                inversed_input_indices < end
            )

            indices_relevant_for_this_mlp = inversed_input_indices[mask]

            embeddings = self.anchor_embeddings[MLP_index]((indices_relevant_for_this_mlp - int(start)).type(torch.IntTensor).to(self.device))
            output_of_MLP = self.mlp[MLP_index](embeddings)
    
            outputs[mask] = output_of_MLP

        return outputs