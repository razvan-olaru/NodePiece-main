import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
import random
import numpy as np
from torch.nn import functional
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Optional, Dict, List
from collections import Counter, defaultdict

from torch import nn
from pykeen.losses import Loss
from pykeen.triples import TriplesFactory
from pykeen.models import Model
from tqdm import tqdm
from nodepiece_tokenizer import NodePiece_Tokenizer


class MLP(nn.Module):
    """
    This class implements an MLP in Pytorch.
    It handles the different layers and parameters of the model.
    """
    
    def __init__(self, n_input, n_hidden, n_classes):
        """
        
        Args:
          n_input: number of input dimensions
          n_hidden: number of hidden dimensions
          n_classes: number of output dimensions
          
        """        

        super().__init__()
        
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        
        self.mlp = nn.Sequential(
            nn.Linear(self.n_input, self.n_hidden),
            nn.ReLU(),
            nn.Linear(self.n_hidden, self.n_classes)
        )
    
    def forward(self, x):
        """
        Performs forward pass of the input.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        """

        out = self.mlp(x)
        
        return out

class NodePieceRotate(Model):

    def __init__(self,
                 tokenizer: NodePiece_Tokenizer = None,
                 triples: TriplesFactory = None,
                 device: torch.device = None,
                 loss: Loss = None,
                 max_paths: int = None,  # max anchors per node
                 subbatch: int = 32,
                 max_seq_len: int = None,  # tied with anchor distances
                 embedding_dim: int = 100,
                 hid_dim: int = 200,  # hidden dim for the hash encoder
                 num_heads: int = 4,  # for Trf
                 num_layers: int = 2,  # for Trf
                 pooler: str = "cat",  # "cat" or "trf"
                 drop_prob: float = 0.1,  # dropout
                 use_distances: bool = True,
                 rel_policy: str = "sum",
                 random_hashes: int = 0,  # for ablations
                 nearest: bool = True,  # use only K nearest anchors per node
                 sample_rels: int = 0,  # size of the relational context
                 ablate_anchors: bool = False,  # for ablations - node hashes will be constructed only from the relational context
                 tkn_mode: str = "path",  # default NodePiece vocabularization strategy
                 clustered_anchors: List = None, 
                 reduced_length: int = None,
                 ):

        super().__init__(
            triples_factory=triples,
            loss=loss,
            predict_with_sigmoid=False,
            automatic_memory_optimization=False,
            preferred_device=device,
        )

        self.pooler = pooler
        self.policy = rel_policy
        self.nearest = nearest
        self.sample_rels = sample_rels
        self.ablate_anchors = ablate_anchors
        self.tkn_mode = tkn_mode

        # cat pooler - concat all anchors+relations in one big vector, pass through a 2-layer MLP
        if pooler == "cat":
            # print(embedding_dim)
            # print(max_paths)
            # print(sample_rels)
            # print(embedding_dim * (max_paths + sample_rels))
            self.set_enc = nn.Sequential(
                nn.Linear(embedding_dim * (max_paths + sample_rels), embedding_dim * 2), nn.Dropout(drop_prob), nn.ReLU(),
                nn.Linear(embedding_dim * 2, embedding_dim)
            ) if not self.ablate_anchors else nn.Sequential(
                nn.Linear(embedding_dim * sample_rels, embedding_dim * 2), nn.Dropout(drop_prob), nn.ReLU(),
                nn.Linear(embedding_dim * 2, embedding_dim)
            )
        # trf pooler - vanilla transformer encoder with mean pooling on top
        elif pooler == "trf":
            ###### Transformer Encoder

            encoder_layer = TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=num_heads,
                dim_feedforward=hid_dim,
                dropout=drop_prob,
            )
            self.set_enc = TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

            self.mlp = nn.Sequential(
                nn.Linear(reduced_length, 64),
                nn.ReLU(),
                nn.Linear(64, embedding_dim)
            )



        self.device = device
        self.loss = loss
        self.triples_factory = triples
        self.tokenizer = tokenizer
        self.random_hashes = random_hashes

        self.subbatch = subbatch
        self.embedding_dim = embedding_dim
        self.real_embedding_dim = embedding_dim // 2  # RotatE interaction assumes vectors conists of two parts: real and imaginary

        self.max_seq_len = max_seq_len
        self.sample_paths = max_paths
        self.use_distances = use_distances

        # pykeen stuff
        self.automatic_memory_optimization = False
        self.tokenizer.token2id[self.tokenizer.NOTHING_TOKEN] = len(tokenizer.token2id) - 1  # TODO this is a bugfix as PathTrfEncoder puts its own index here


        self.clustered_anchors = clustered_anchors
        self.reduced_length = reduced_length
        # self.mlp = MLP(self.reduced_length, 64, self.embedding_dim)

        print("NN EMBEDDINGS DIMENSIONS: ", len(tokenizer.token2id), embedding_dim)
        print("New Shape: ", len(self.clustered_anchors), self.reduced_length)
        print("Padding token: ", tokenizer.PADDING_TOKEN, self.tokenizer.token2id, self.tokenizer.token2id[tokenizer.PADDING_TOKEN])

        # if self.tokenizer.PADDING_TOKEN in self.clustered_anchors:
        #     self.anchor_embeddings = nn.Embedding(len(self.clustered_anchors), embedding_dim=self.reduced_length, padding_idx=self.clustered_anchors[self.tokenizer.PADDING_TOKEN])
        # else:
        #     self.anchor_embeddings = nn.Embedding(len(self.clustered_anchors), embedding_dim=self.reduced_length)
        
        self.anchor_embeddings = nn.Embedding(len(tokenizer.token2id), embedding_dim=reduced_length, padding_idx=self.tokenizer.token2id[tokenizer.PADDING_TOKEN])

        # self.anchor_embeddings = nn.Embedding(len(tokenizer.token2id), embedding_dim=embedding_dim, padding_idx=self.tokenizer.token2id[tokenizer.PADDING_TOKEN])
        
        self.relation_embeddings = nn.Embedding(self.triples_factory.num_relations + 1, embedding_dim=embedding_dim, padding_idx=self.triples_factory.num_relations)
        self.dist_emb = nn.Embedding(self.max_seq_len, embedding_dim=embedding_dim)
        self.entity_embeddings = None

        # now fix anchors per node for each node in a graph, either deterministically or randomly
        # we do it mostly for speed reasons, although this process can happen during the forward pass either
        if self.random_hashes == 0:
            if self.tkn_mode != "bfs":
                # DETERMINISTIC strategy
                if not self.nearest:
                    # subsample paths, need to align them with distances
                    sampled_paths = {
                        entity: random.sample(paths, k=min(self.sample_paths, len(paths)))
                        for entity, paths in self.tokenizer.vocab.items()
                    }
                elif self.nearest:
                    # sort paths by length first and take K of them
                    sampled_paths = {
                        entity: sorted(paths, key=lambda x: len(x))[:min(self.sample_paths, len(paths))]
                        for entity, paths in self.tokenizer.vocab.items()
                    }
                    self.max_seq_len = max(len(path) for k,v in sampled_paths.items() for path in v)
                    print(f"Changed max seq len from {max_seq_len} to {self.max_seq_len} after keeping {self.sample_paths} shortest paths")

                hashes = [
                    [self.tokenizer.token2id[path[0]] for path in paths] + [self.tokenizer.token2id[tokenizer.PADDING_TOKEN]]*(self.sample_paths - len(paths))
                    for entity, paths in sampled_paths.items()
                ]
                distances = [
                    [len(path)-1 for path in paths] + [0] *(self.sample_paths - len(paths))
                    for entity, paths in sampled_paths.items()
                ]
                total_paths = [
                    [len(paths)]
                    for entity, paths in sampled_paths.items()
                ]
            else:
                # only nearest neighbors mode
                if not self.nearest:
                    raise NotImplementedError("bfs mode works only with -nn True")
                hashes = [
                    [self.tokenizer.token2id[token] for token in vals['ancs'][:min(self.sample_paths, len(vals['ancs']))]] + [self.tokenizer.token2id[tokenizer.PADDING_TOKEN]] * (self.sample_paths - len(vals['ancs']))
                    for entity, vals in self.tokenizer.vocab.items()
                ]
                distances = [
                    [d for d in vals['dists'][:min(self.sample_paths, len(vals['dists']))]] + [0] * (self.sample_paths - len(vals['dists']))
                    for entity, vals in self.tokenizer.vocab.items()
                ]
                total_paths = distances
                self.max_seq_len = max([d for row in distances for d in row])
                print(f"Changed max seq len from {max_seq_len} to {self.max_seq_len} after keeping {self.sample_paths} shortest paths")

            self.hashes = torch.tensor(hashes, dtype=torch.long, device=self.device)
            self.distances = torch.tensor(distances, dtype=torch.long, device=self.device)
            self.total_paths = torch.tensor(total_paths, dtype=torch.long, device=self.device)

        else:
            # RANDOM strategy
            # in this case, we bypass distances and won't use relations in the encoder
            print("NN EMBEDDINGS 2: ", self.random_hashes, embedding_dim)
            self.anchor_embeddings = nn.Embedding(self.random_hashes, embedding_dim=embedding_dim)
            hashes = [
                random.sample(list(range(random_hashes)), self.sample_paths)
                for i in range(triples.num_entities)
            ]

            self.hashes = torch.tensor(hashes, dtype=torch.long, device=self.device)
            self.distances = torch.zeros((triples.num_entities, self.sample_paths), dtype=torch.long, device=self.device)
            self.total_paths = torch.zeros((triples.num_entities, 1), dtype=torch.long, device=self.device)

        # creating the relational context of M unique outgoing relation types for each node
        if self.sample_rels > 0:
            pad_idx = self.triples_factory.num_relations
            e2r = defaultdict(set)
            for row in self.triples_factory.mapped_triples:
                e2r[row[0].item()].add(row[1].item())
            len_stats = [len(v) for k,v in e2r.items()]
            print(f"Unique relations per node - min: {min(len_stats)}, avg: {np.mean(len_stats)}, 66th perc: {np.percentile(len_stats, 66)}, max: {max(len_stats)} ")
            unique_1hop_relations = [
                random.sample(e2r[i], k=min(self.sample_rels, len(e2r[i]))) + [pad_idx] * (self.sample_rels-min(len(e2r[i]), self.sample_rels))
                for i in range(self.triples_factory.num_entities)
            ]
            self.unique_1hop_relations = torch.tensor(unique_1hop_relations, dtype=torch.long, device=self.device)


    def _reset_parameters_(self):

        if self.pooler != "avg":
            for module in self.set_enc.modules():
                if module is self:
                    continue
                if hasattr(module, "reset_parameters"):
                    module.reset_parameters()

            if self.pooler == "mlp":
                for module in self.set_dec.modules():
                    if module is self:
                        continue
                    if hasattr(module, "reset_parameters"):
                        module.reset_parameters()


        torch.nn.init.xavier_uniform_(self.anchor_embeddings.weight)
        torch.nn.init.xavier_uniform_(self.dist_emb.weight)


        # if self.random_hashes == 0:
        #     with torch.no_grad():
        #         if self.tokenizer.PADDING_TOKEN in self.clustered_anchors:
        #             self.anchor_embeddings.weight[self.clustered_anchors[self.tokenizer.PADDING_TOKEN]] = torch.zeros(self.reduced_length)
        #         self.dist_emb.weight[0] = torch.zeros(self.embedding_dim)

        if self.random_hashes == 0:
            with torch.no_grad():
                print("Random Hashes")
                self.anchor_embeddings.weight[self.tokenizer.token2id[self.tokenizer.PADDING_TOKEN]] = torch.zeros(self.reduced_length)
                self.dist_emb.weight[0] = torch.zeros(self.embedding_dim)

        # if self.random_hashes == 0:
        #     with torch.no_grad():
        #         print("Random Hashes")
        #         self.anchor_embeddings.weight[self.tokenizer.token2id[self.tokenizer.PADDING_TOKEN]] = torch.zeros(self.embedding_dim)
        #         self.dist_emb.weight[0] = torch.zeros(self.embedding_dim)


        # for RotatE: phases randomly between 0 and 2 pi
        phases = 2 * np.pi * torch.rand(self.num_relations, self.real_embedding_dim, device=self.device)
        relations = torch.stack([torch.cos(phases), torch.sin(phases)], dim=-1).detach()
        assert torch.allclose(torch.norm(relations, p=2, dim=-1), phases.new_ones(size=(1, 1)))
        self.relation_embeddings.weight.data[:-1] = relations.view(self.num_relations, self.embedding_dim)
        self.relation_embeddings.weight.data[-1] = torch.zeros(self.embedding_dim)


    def post_parameter_update(self):  # noqa: D102

        # Make sure to call super first
        super().post_parameter_update()

        # Normalize relation embeddings
        rel = self.relation_embeddings.weight.data.view(self.num_relations+1, self.real_embedding_dim, 2)
        rel = functional.normalize(rel, p=2, dim=-1)
        self.relation_embeddings.weight.data = rel.view(self.num_relations+1, self.embedding_dim)

        self.entity_embeddings = None


    def pool_anchors(self, anc_embs: torch.FloatTensor, mask: Optional[torch.BoolTensor] = None) -> torch.FloatTensor:
        """
        input shape: (bs, num_anchors + relational_context, emb_dim)
        output shape: (bs, emb_dim)
        """

        if self.pooler == "cat":
            anc_embs = anc_embs.view(anc_embs.shape[0], -1)
            pooled = self.set_enc(anc_embs) if self.sample_paths != 1 else anc_embs
        elif self.pooler == "trf":         ###HERE
            pooled = self.set_enc(anc_embs.transpose(1, 0))  # output shape: (seq_len, bs, dim)
            pooled = pooled.mean(dim=0)  # output shape: (bs, dim)

        return pooled


    def encode_by_index(self, entities: torch.LongTensor) -> torch.FloatTensor:

        # take a node index and find its NodePiece hash

        # print("Entities: ", entities)
        hashes, dists, ids = self.hashes[entities], self.distances[entities], self.total_paths[entities]

        print("Input Embedding Layer: ", hashes.shape)
        print("Embedding Layer: ", self.anchor_embeddings)
        anc_embs = self.anchor_embeddings(hashes)
        print("Output Embedding Layer: ", anc_embs.shape)

        # print("MLP Layer: ", self.mlp)

        # batch = torch.tensor(np.random.rand(10, 40), dtype=torch.float, device=self.device)
        # self.batch = torch.rand((10, 40), device=self.device)

        # self.hashes = torch.tensor((10, 40), dtype=torch.long, device=self.device)
        # print("Self Hashes: ", self.hashes)

        anc_embs = self.mlp(anc_embs)
        print("Output MLP Layer: ", anc_embs.shape, '\n')


        mask = None

        if self.use_distances:
            dist_embs = self.dist_emb(dists)
            anc_embs += dist_embs

        # for ablations: drop anchors entirely
        if self.ablate_anchors:
            anc_embs = torch.tensor([], device=self.device)

        # add relational context (if its size > 0 )
        if self.sample_rels > 0:
            rels = self.unique_1hop_relations[entities]  # (bs, rel_sample_size)
            rels = self.relation_embeddings(rels)   # (bs, rel_sample_size, dim)
            anc_embs = torch.cat([anc_embs, rels], dim=1)  # (bs, ancs+rel_sample_size, dim)

        anc_embs = self.pool_anchors(anc_embs, mask=mask)  # (bs, dim)
        print("ANOTHER ANC EMBS: ", anc_embs.shape)

        return anc_embs


    def get_all_representations(self):

        # materialize embeddings for all nodes in a graph for scoring

        temp_embs = torch.zeros((len(self.hashes), self.embedding_dim), dtype=torch.float, device=self.device)

        vocab_keys = list(range(len(self.hashes)))
        for i in tqdm(range(0, len(self.hashes), self.subbatch)):
            entities = torch.tensor(vocab_keys[i: i+self.subbatch], dtype=torch.long, device=self.device)
            embs = self.encode_by_index(entities)
            temp_embs[i: i + self.subbatch, :] = embs
        print("TEMP EMBS: ", temp_embs.shape, '\n')
        return temp_embs


    @staticmethod
    def pairwise_interaction_function(
            h: torch.FloatTensor,
            r: torch.FloatTensor,
    ) -> torch.FloatTensor:

        # Decompose into real and imaginary part
        h_re = h[..., 0]
        h_im = h[..., 1]
        r_re = r[..., 0]
        r_im = r[..., 1]

        # Rotate (=Hadamard product in complex space).
        rot_h = torch.cat(
            [
                h_re * r_re - h_im * r_im,
                h_re * r_im + h_im * r_re,
            ],
            dim=-1,
        )

        return rot_h

    @staticmethod
    def interaction_function(
        h: torch.FloatTensor,
        r: torch.FloatTensor,
        t: torch.FloatTensor,
    ) -> torch.FloatTensor:

        # Decompose into real and imaginary part
        h_re = h[..., 0]
        h_im = h[..., 1]
        r_re = r[..., 0]
        r_im = r[..., 1]

        # Rotate (=Hadamard product in complex space).
        rot_h = torch.stack(
            [
                h_re * r_re - h_im * r_im,
                h_re * r_im + h_im * r_re,
            ],
            dim=-1,
        )
        # Workaround until https://github.com/pytorch/pytorch/issues/30704 is fixed
        diff = rot_h - t
        scores = -torch.norm(diff.view(diff.shape[:-2] + (-1,)), dim=-1)

        return scores

    def score_hrt(self, hrt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102


        # when training with large # of neg samples the hrt_batch size can be too big to fit into memory, so chunk it
        if hrt_batch.shape[0] <= self.subbatch or self.subbatch == 0:
            # Get embeddings
            print("hrt_batch: ", hrt_batch[:, 0].shape)
            h = self.encode_by_index(hrt_batch[:, 0]).view(-1, self.real_embedding_dim, 2)
            r = self.relation_embeddings(hrt_batch[:, 1]).view(-1, self.real_embedding_dim, 2)
            t = self.encode_by_index(hrt_batch[:, 2]).view(-1, self.real_embedding_dim, 2)

            # h = self.encode_by_index(hrt_batch[:, 0]).view(-1, self.reduced_length//2, 2)
            # r = self.relation_embeddings(hrt_batch[:, 1]).view(-1, self.reduced_length//2, 2)
            # t = self.encode_by_index(hrt_batch[:, 2]).view(-1, self.reduced_length//2, 2)

            # Compute scores
            scores = self.interaction_function(h=h, r=r, t=t).view(-1, 1)
        else:
            scores = torch.zeros((hrt_batch.shape[0], 1), dtype=torch.float, device=hrt_batch.device)
            for i in range(0, hrt_batch.shape[0], self.subbatch):
                h = self.encode_by_index(hrt_batch[i: i+self.subbatch, 0]).view(-1, self.real_embedding_dim, 2)
                r = self.relation_embeddings(hrt_batch[i: i+self.subbatch, 1]).view(-1, self.real_embedding_dim, 2)
                t = self.encode_by_index(hrt_batch[i: i+self.subbatch, 2]).view(-1, self.real_embedding_dim, 2)

                # Compute scores
                scores[i: i+self.subbatch] = self.interaction_function(h=h, r=r, t=t).view(-1, 1)

        print("SCORES? : ", scores.shape)
        return scores


    def score_t(self, hr_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102

        # Get embeddings
        h = self.encode_by_index(hr_batch[:, 0]).view(-1, 1, self.real_embedding_dim, 2)
        r = self.relation_embeddings(hr_batch[:, 1]).view(-1, 1, self.real_embedding_dim, 2)

        # Rank against all entities, don't use hard negs, EXPENSIVE
        t = self.get_all_representations().view(1, -1, self.real_embedding_dim, 2)

        # Compute scores
        if self.subbatch == 0:
            scores = self.interaction_function(h=h, r=r, t=t)
        else:
            scores = torch.zeros((hr_batch.shape[0], t.shape[1]), dtype=torch.float, device=hr_batch.device)
            for i in tqdm(range(0, t.shape[1], self.subbatch)):
                temp_scores = self.interaction_function(h=h, r=r, t=t[:, i:i+self.subbatch, :, :])
                scores[:, i:i+self.subbatch] = temp_scores

        return scores

    def score_h(self, rt_batch: torch.LongTensor) -> torch.FloatTensor:  # noqa: D102

        # Get embeddings
        r = self.relation_embeddings(rt_batch[:, 0]).view(-1, 1, self.real_embedding_dim, 2)
        t = self.encode_by_index(rt_batch[:, 1]).view(-1, 1, self.real_embedding_dim, 2)

        r_inv = torch.stack([r[:, :, :, 0], -r[:, :, :, 1]], dim=-1)

        # Rank against all entities
        h = self.get_all_representations().view(1, -1, self.real_embedding_dim, 2)

        # Compute scores
        if self.subbatch == 0:
            scores = self.interaction_function(h=t, r=r_inv, t=h)
        else:
            scores = torch.zeros((rt_batch.shape[0], t.shape[1]), dtype=torch.float, device=rt_batch.device)
            for i in tqdm(range(0, t.shape[1], self.subbatch)):
                temp_scores = self.interaction_function(h=t, r=r_inv, t=h[:, i:i+self.subbatch, :, :])
                scores[:, i:i+self.subbatch] = temp_scores


        return scores