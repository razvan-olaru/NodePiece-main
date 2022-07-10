import imp

import torch
import torch.nn as nn


class VariableSizedEmbedding(nn.Module):
    """TODO: complete documentation
        This is similar to the torch embedding moduls. However, internally only the number of dimensions for each embedding varies depending on how it was specified.
        To get the same dimension out when performing a lookup, an MLP is used to scale up the embeddings.
    """

    def __init__(self, embedding_sizes: torch.Tensor, embedding_dimension: int, hid_dim: int) -> None:
        """Create by specifiying the size for each of the embeddings, and the final (output) embedding size.
        So, if you need 5 embeddings with sizes 5, 10, 5, 10, 20 (in that order), and a final embedding size of 50, you create is using:
        VariableSizedEmbedding(torch.LongTensor([5, 10, 5, 10, 2]), 50)
        VariableSizedEmbedding(torch.LongTensor([20, 20, 20, 10, 20, 40, ...]), 200)
        """
        super().__init__()

        # embedding size = a list of the reduced embedding sized for all anchors
        # embedding dimension = the higher dimension we want to transform to, ex 200
        self.n_hidden = hid_dim
        self.embedding_dimension = embedding_dimension
        print("Embedding Sizes: ", embedding_sizes)

        # first we sort the embedding sizes and REMEMBER HOW THE SORTING WENT! Check the documentation on indices.
        values, self.indices = torch.sort(embedding_sizes)
        print("Sorted values: ", values)
        print("Their indices: ", self.indices)

        # We need to remember in which location in our sorted list each item now is, to do this, we keep the indices of the indices
        self.inverse_indices = torch.argsort(self.indices)
        print("Self Inverse Indices: ", self.inverse_indices, self.inverse_indices.shape)

        # now we need to know how many of each we have
        unique_sizes, counts = torch.unique_consecutive(
            values, return_inverse=False, return_counts=True, dim=None)
        print("Unique sizes: ", unique_sizes, counts)
        self.unique_sizes = unique_sizes

        # TODO: for each of these unique sizes, make a normal Embedding, contianing count embeddings, store them in a list
        # make sure to store them in self in a torch.nn.ParameterList
        # We create an Embedding for each cluster. Put them all in ModuleList()
        # PART 1 OF MODEL: nn.Embeddings
        self.anchor_embeddings = nn.ModuleList()
        for i in range(len(unique_sizes)):
            self.anchor_embeddings.append(nn.Embedding(int(counts[i]), embedding_dim=int(unique_sizes[i]), padding_idx=0))

        # TODO: create the requered number of MLPs. Also store these in a torch.nn.ParameterList
        # We create an MLP for each cluster. Put them all in ModuleList()
        # PART 2 OF MODEL: MLP
        self.mlp = nn.ModuleList()
        for i in range(len(unique_sizes)):
            self.mlp.append(nn.Sequential(
                nn.Linear(int(unique_sizes[i]), self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.embedding_dimension)
            ))

        # now we rememebr the cumulative counts. They are or offsets needed for lookup in the list later
        cummulative_counts = torch.cumsum(counts, dim=0)
        self.start_offsets = torch.cat(
            (torch.Tensor([0]), cummulative_counts[:-1]))
        self.end_offsets = cummulative_counts
        print("Cummulative counts: ", cummulative_counts)
        # Start offsets: Start index of cluster i; End offsets: End index of cluster i; for all i
        print("Start offset, end offset: ", self.start_offsets, self.end_offsets, '\n')

    def reset_parameters(self) -> None:
        # TODO: reset all MLPs
        for i in range(len(self.unique_sizes)):
            torch.nn.init.xavier_uniform_(self.anchor_embeddings[i].weight)
        for i in range(len(self.unique_sizes)):
            torch.nn.init.xavier_uniform_(self.mlp[i][0].weight)
            torch.nn.init.xavier_uniform_(self.mlp[i][2].weight)

    def forward(self, input: torch.Tensor):
        """Lookup the embeddings"""

        # 1 get the reversed indices
        print("Input: ", input, self.inverse_indices.shape)
        # Extra padding token replaced with index 0 (padding id) to not throw out of bounds error
        input[input == self.inverse_indices.shape[0]] = 0

        inversed_input_indices = self.inverse_indices[input]
        print("Inverse Input Indices: ", inversed_input_indices)

        # iterate of the different sizes and apply the right MLP

        # in principle we do not need to zero the tensor, because we overwrite it below but this seems useful for debugging.
        # TODO check the dimensions of this operation
        outputs = torch.zeros(input.shape[0], input.shape[1], self.embedding_dimension)

        for MLP_index, (start, end) in enumerate(zip(self.start_offsets, self.end_offsets)):
            print("MLP: ", MLP_index, start, end)
            mask = torch.logical_and(
                inversed_input_indices >= start,
                inversed_input_indices < end
            )

            print("MASK: ", mask)
            indices_relevant_for_this_mlp = inversed_input_indices[mask]

            # TODO: lookup these indices in the relevant torch.Embedding, note here that we have to subtract 'start' from the indices!
            # Forward to the Embeddings:
            embeddings = self.anchor_embeddings[MLP_index]((indices_relevant_for_this_mlp - int(start)).type(torch.IntTensor))

            # # TODO: scale the embeddings using the correct MLP
            # Forward to the MLP:
            output_of_MLP = self.mlp[MLP_index](embeddings)

            # # TODO: insert the outputs of the MLP to the outputs
            outputs[mask] = output_of_MLP

        print("FINAL OUTPUT: ", outputs.shape)

        return outputs