from pykeen.datasets import FB15k237, WN18RR, YAGO310
from pykeen.utils import resolve_device
from pykeen.losses import BCEWithLogitsLoss, SoftplusLoss, MarginRankingLoss, NSSALoss
from pykeen.training import SLCWATrainingLoop, LCWATrainingLoop
from pykeen.trackers import WANDBResultTracker
from pykeen.evaluation import RankBasedEvaluator
from pykeen.sampling import BasicNegativeSampler
from pykeen.stoppers import EarlyStopper
from pykeen.models import RotatE
from torch.optim import Adam

from pykeen105.nodepiece_rotate import NodePieceRotate
from loops.filtered_sampling_loop import FilteredSLCWATrainingLoop
from datasets.codex import CoDExLarge
from nodepiece_tokenizer import NodePiece_Tokenizer

from pykeen105.relation_rank_evaluator import RelationPredictionRankBasedEvaluator
from pykeen105.negative_sampler import RelationalNegativeSampler

import torch
import click
import numpy as np
import wandb
import random
import pickle

from networkx import degree_centrality
from sklearn.cluster import KMeans
import sys
sys.path.insert(0, 'C:/Users/razva/Desktop/NodePiece-main/nc/')


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


DEFAULT_CONFIG = {
    'BATCH_SIZE': 512,
    'CORRUPTION_POSITIONS': [0, 2],
    'DATASET': 'wd50k',
    'DEVICE': 'cpu',
    'EMBEDDING_DIM': 50,
    'FEATURE_DIM': 1024,
    'ENT_POS_FILTERED': True,
    'EPOCHS': 1000,
    'EVAL_EVERY': 1,
    'LEARNING_RATE': 0.0002,
    'MARGIN_LOSS': 5,
    'MAX_QPAIRS': 3,
    'MODEL_NAME': 'stare',
    'NUM_FILTER': 5,
    'RUN_TESTBENCH_ON_TRAIN': True,
    'SAVE': False,
    'SELF_ATTENTION': 0,
    'STATEMENT_LEN': 3,
    'USE_TEST': False,
    'WANDB': False,
    'LABEL_SMOOTHING': 0.0,
    'OPTIMIZER': 'adam',
    'CLEANED_DATASET': True,
    'NUM_RUNS': 1,

    'GRAD_CLIPPING': True,
    'LR_SCHEDULER': False,

    'CL_TASK': 'so',  # so or full
    'DS_TYPE': 'transductive',  # transductive or inductive
    'IND_V': 'v1',  # v1 or v2
    'SWAP': False,
    'TR_RATIO': 1.0,
    'VAL_RATIO': 1.0,
    'RANDPERM': False,

    'PYG_DATA': True,
    'USE_FEATURES': True,
    'WEIGHT_LOSS': False,

    # NodePiece args
    'SUBBATCH': 5000,
    'TOKENIZE': False,
    'NUM_ANCHORS': 500,
    'MAX_PATHS': 20,
    'NEAREST': True,
    'POOLER': 'cat',
    'SAMPLE_RELS': 5,
    'RANDOM_HASHES': 0,
    'MAX_PATH_LEN': 0,
    'USE_DISTANCES': True,
    'T_LAYERS': 2,
    'T_HEADS': 4,
    'T_HIDDEN': 512,
    'T_DROP': 0.1,
    'NO_ANC': False,
}


STAREARGS = {
    'LAYERS': 2,
    'N_BASES': 0,
    'GCN_DIM': 80,
    'GCN_DROP': 0.1,
    'HID_DROP': 0.3,
    'HID_DROP2': 0.1,
    'FEAT_DROP': 0.3,
    'BIAS': False,
    'OPN': 'rotate',
    'TRIPLE_QUAL_WEIGHT': 0.8,
    'QUAL_AGGREGATE': 'sum',  # or concat or mul
    'QUAL_OPN': 'rotate',
    'QUAL_N': 'sum',  # or mean
    'SUBBATCH': 0,
    'QUAL_REPR': 'sparse',  # sparse or full
    'ATTENTION': False,
    'ATTENTION_HEADS': 4,
    'ATTENTION_SLOPE': 0.2,
    'ATTENTION_DROP': 0.1,

    # LRGA
    'LRGA': False,
    'LRGA_K': 50,
    'LRGA_DROP': 0.2

}

DEFAULT_CONFIG['STAREARGS'] = STAREARGS


@click.command()
@click.option('-embedding', '--embedding-dimension', type=int, default=100)  # embedding dim for anchors and relations
@click.option('-loss', '--loss_fc', type=str, default='nssal')
@click.option('-loop', '--loop', type=str, default='slcwa')  # slcwa - negative sampling, lcwa - 1-N scoring
@click.option('-trf_hidden', '--transformer-hidden-dim', type=int, default=512)
@click.option('-trf_heads', '--transformer-num-heads', type=int, default=4)
@click.option('-trf_layers', '--transformer-layers', type=int, default=2)
@click.option('-trf_drop', '--transformer-dropout', type=float, default=0.1)
@click.option('-b', '--batch-size', type=int, default=512)
@click.option('-epochs', '--num-epochs', type=int, default=1000)
@click.option('-lr', '--learning-rate', type=float, default=0.0005)
@click.option('-wandb', '--enable_wandb', type=bool, default=False)
@click.option('-anchors', '--topk_anchors', type=int, default=500)  # how many total anchors to use
@click.option('-data', '--dataset_name', type=str, default='wn18rr')
@click.option('-anc_deg', '--strategy_degree', type=float, default=0.4)  # % of anchors selected based on top node degree
@click.option('-anc_betw', '--strategy_betweenness', type=float, default=0.0)  # disabled
@click.option('-anc_ppr', '--strategy_pagerank', type=float, default=0.4)  # % of anchors selected based on top PPR
@click.option('-anc_rand', '--strategy_random', type=float, default=0.2)  # % of randomly selected anchors
@click.option('-sp', '--k_shortest_paths', type=int, default=0)  # when mining anchors per node - keep K closest
@click.option('-rp', '--k_random_paths', type=int, default=0)  # when mining anchors per node - keep K random
@click.option('-eval_every', '--eval_every', type=int, default=1)
@click.option('-mtype', '--model_type', type=str, default="nodepiece")  # or "rotate" for the baseline
@click.option('-ft_maxp', '--ft_max_paths', type=int, default=100)  # max anchor per node, should be <= total N of anchors
@click.option('-anc_dist', '--use_anchor_distances', type=bool, default=True)  # whether to add anchor distances
@click.option('-margin', '--margin', type=float, default=15)
@click.option('-max_seq_len', '--max_seq_len', type=int, default=0)
@click.option('-pool', '--pooling', type=str, default="cat")  # available encoders: "cat" or "trf"
@click.option('-subbatch', '--trf_subbatch', type=int, default=3000)
@click.option('-negs', '--num_negatives_ent', type=int, default=1)  # number of negative samples when training LP in sLCWA
@click.option('-negs-rel', '--num_negatives-rel', type=int, default=1)  # Optional: number of negative relations when training RP in sLCWA
@click.option('-rel-prediction', '--rel-prediction', type=bool, default=False)  # swtich to the Relation Prediction task (RP)
@click.option('-smoothing', '--lbl_smoothing', type=float, default=0.0)  # label smoothing in the 1-N setup
@click.option('-relpol', '--rel_policy', type=str, default="sum")
@click.option('-filtered_sampling', '--filtered_sampling', type=bool, default=False)
@click.option('-rand_hashes', '--random_hashing', type=int, default=0)  # for ablations: use only random numbers as hashes
@click.option('-nn', '--nearest_neighbors', type=bool, default=True)  # use only nearest anchors per node
@click.option('-sample_rels', '--sample_rels', type=int, default=4)  # size of the relational context M
@click.option('-anchor_eye', '--anchor_eye', type=bool, default=False)  # anchors in their own hashes will have their index at the frist place
@click.option('-tkn_mode', '--tkn_mode', type=str, default="path")  # mining paths in iGRAPH
@click.option('-no_anc', '--ablate_anchors', type=bool, default=False)  # don't use any anchors in hashes, keep only the relational context

    

def main(
        embedding_dimension,
        loss_fc,
        loop,
        transformer_hidden_dim: int,
        transformer_num_heads: int,
        transformer_layers: int,
        transformer_dropout: float,
        batch_size: int,
        num_epochs: int,
        learning_rate: float,
        enable_wandb: bool,
        topk_anchors: int,
        dataset_name: str,
        strategy_degree: float,
        strategy_betweenness: float,
        strategy_pagerank: float,
        strategy_random: float,
        k_shortest_paths: int,
        k_random_paths: int,
        eval_every: int,
        model_type: str,
        ft_max_paths: int,
        use_anchor_distances: bool,
        margin: float,
        max_seq_len: int,
        pooling: str,
        trf_subbatch: int,
        num_negatives_ent: int,
        num_negatives_rel: int,
        rel_prediction: bool,
        lbl_smoothing: float,
        rel_policy: str,
        filtered_sampling: bool,
        random_hashing: int,
        nearest_neighbors: bool,
        sample_rels: int,
        anchor_eye: bool,
        tkn_mode: str,
        ablate_anchors: bool
):
    # Standard dataset loading procedures, inverses are necessary for reachability of nodes
    if dataset_name == 'fb15k237':
        dataset = FB15k237(create_inverse_triples=True)
    elif dataset_name == 'wn18rr':
        dataset = WN18RR(create_inverse_triples=True)
    elif dataset_name == "yago":
        dataset = YAGO310(create_inverse_triples=True)
    elif dataset_name == "codex_l":
        dataset = CoDExLarge(create_inverse_triples=True)

    # if we're in the RP task - change the evaluator
    if rel_prediction:
        evaluator_type = RelationPredictionRankBasedEvaluator
    else:
        evaluator_type = RankBasedEvaluator

    # sampling even harder negatives - turned off by default
    if filtered_sampling:
        negative_sampler_cls = RelationalNegativeSampler
        negative_sampler_kwargs = dict(num_negs_per_pos=num_negatives_ent, num_negs_per_pos_rel=num_negatives_rel,
                                       dataset_name=dataset_name)
        loop_type = FilteredSLCWATrainingLoop
    else:
        negative_sampler_cls = BasicNegativeSampler
        negative_sampler_kwargs = dict(num_negs_per_pos=num_negatives_ent)
        loop_type = SLCWATrainingLoop


    training_triples_factory = dataset.training

    # Now let's create a NodePiece tokenizer
    kg_tokenizer = NodePiece_Tokenizer(triples=training_triples_factory,
                                anchor_strategy={
                                    "degree": strategy_degree,
                                    "betweenness": strategy_betweenness,
                                    "pagerank": strategy_pagerank,
                                    "random": strategy_random
                                },
                                num_anchors=topk_anchors, dataset_name=dataset_name, limit_shortest=k_shortest_paths,
                                add_identity=anchor_eye, mode=tkn_mode, limit_random=k_random_paths)

    ################################################################## START
    project = True

    if project == True:

        # Step 0: Get Graph
        with open('data/wn18rr_30_anchors_30_paths_d0.4_b0.0_p0.4_r0.2_pykeen_20sp.pkl', 'rb') as f:
            data = pickle.load(f)

        top_entities, _, _, graph = data

        # Step 1: Degree Centrality
        G = graph.to_networkx()
        G_degree_centrality = degree_centrality(G)

        # Step 1.2: Clustering
        degree_values = np.array(list(G_degree_centrality.values())).reshape(-1, 1)

        # Apply Elbow rule to find optimal k
        sse = []
        for k in range(1, 10):
            kmeans = KMeans(n_clusters=k).fit(degree_values)
            sse.append(kmeans.inertia_) # Inertia: Sum of distances of samples to their closest cluster center
        
        sse_diff = [sse[i] - sse[i+1] for i in range(len(sse)-1)]
        best_k = [i+1 for i,v in enumerate(sse_diff) if v < 0.0005][0]
        print("Best k: ", best_k)

        # Elbow Rule Plot:
        # plt.figure()
        # plt.plot(list(range(1, len(sse)+1)), sse)
        # plt.xlabel("Number of cluster")
        # plt.ylabel("SSE")
        # plt.show()

        # Apply kmeans with best k
        kmeans = KMeans(n_clusters=best_k).fit(degree_values)

        # Creates a dict: "anchor_id" : [anchor's cluster id, number of anchors in that cluster]
        anchors_clusters = {}
        for i in range(len(top_entities)):
            current_cluster = kmeans.labels_[top_entities][i]
            anchors_clusters[top_entities[i]] = [current_cluster, list(kmeans.labels_[top_entities]).count(current_cluster)]

        # Step 2: Get the new reduced embedding length
        anchors_reduced_length= {}
        for i in range(len(top_entities)):
            anchors_reduced_length[top_entities[i]] = int(embedding_dimension / anchors_clusters[top_entities[i]][1])

        anchors_reduced_length_sorted = sorted(anchors_reduced_length.items())
        anchors_reduced_embeddings = torch.Tensor([x[1] for x in anchors_reduced_length_sorted])

    ################################################################## END

        device = resolve_device()
    

        # cater for corner cases when user-input max seq len is incorrect
        if max_seq_len == 0 or max_seq_len != (kg_tokenizer.max_seq_len+3):
            max_seq_len = kg_tokenizer.max_seq_len + 3  # as in the PathTrfEncoder, +1 CLS, +1 PAD, +1 LP tasks
            print(f"Set max_seq_len to{max_seq_len}")

        # for stability
        kg_tokenizer.token2id[kg_tokenizer.NOTHING_TOKEN] = kg_tokenizer.vocab_size

        # selecting the loss function
        if loss_fc == "softplus":
            ft_loss = SoftplusLoss()
        elif loss_fc == "bce":
            ft_loss = BCEWithLogitsLoss()
        elif loss_fc == "mrl":
            ft_loss = MarginRankingLoss(margin=margin)
        elif loss_fc == "nssal":
            ft_loss = NSSALoss(margin=margin)

        train_factory = dataset.training
        validation_factory = dataset.validation

        if model_type == "baseline":
            finetuning_model = RotatE(embedding_dim=embedding_dimension // 2, triples_factory=train_factory,
                                loss=ft_loss, automatic_memory_optimization=False, preferred_device=device)
            optimizer = Adam(params=finetuning_model.parameters(), lr=learning_rate)
            print(f"Vanilla rotate created, Number of params: {sum(p.numel() for p in finetuning_model.parameters())}")
            ft_loop = SLCWATrainingLoop(model=finetuning_model, optimizer=optimizer)

        else:
    
            finetuning_model = NodePieceRotate(embedding_dim=embedding_dimension, device=device, loss=ft_loss,
                                            triples=train_factory, max_paths=ft_max_paths, subbatch=trf_subbatch,
                                            max_seq_len=max_seq_len, tokenizer=kg_tokenizer, pooler=pooling,
                                            hid_dim=transformer_hidden_dim, num_heads=transformer_num_heads,
                                            use_distances=use_anchor_distances, num_layers=transformer_layers, drop_prob=transformer_dropout,
                                            rel_policy=rel_policy, random_hashes=random_hashing, nearest=nearest_neighbors,
                                            sample_rels=sample_rels, ablate_anchors=ablate_anchors, tkn_mode=tkn_mode,
                                            anchors_reduced_embeddings = anchors_reduced_embeddings)

            optimizer = Adam(params=finetuning_model.parameters(), lr=learning_rate)
            print(f"Number of params: {sum(p.numel() for p in finetuning_model.parameters())}")
            if loop == "slcwa":
                ft_loop = loop_type(model=finetuning_model, optimizer=optimizer, negative_sampler_cls=negative_sampler_cls,
                                            negative_sampler_kwargs=negative_sampler_kwargs)
            else:
                ft_loop = LCWATrainingLoop(model=finetuning_model, optimizer=optimizer)

        # add the results tracker if requested
        if enable_wandb:
            project_name = "NodePiece_LP"
            if rel_prediction:
                project_name += "_RP"

            tracker = WANDBResultTracker(project=project_name, group=None, settings=wandb.Settings(start_method='fork'))
            tracker.wandb.config.update(click.get_current_context().params)
        else:
            tracker = None

        valid_evaluator = evaluator_type()
        valid_evaluator.batch_size = 256

        # we don't actually use the early stopper here by setting the patience to 1000
        early_stopper = EarlyStopper(
            model=finetuning_model,
            relative_delta=0.0005,
            evaluation_triples_factory=validation_factory,
            frequency=eval_every,
            patience=1000,
            result_tracker=tracker,
            evaluation_batch_size=256,
            evaluator=valid_evaluator,
        )

        # Train LP / RP
        if loop == "lcwa":
            ft_loop.train(num_epochs=num_epochs, batch_size=batch_size, result_tracker=tracker,
                        stopper=early_stopper, label_smoothing=lbl_smoothing)
        else:
            ft_loop.train(num_epochs=num_epochs, batch_size=batch_size, result_tracker=tracker,
                        stopper=early_stopper)

        # run the final test eval
        test_evaluator = evaluator_type()
        test_evaluator.batch_size = 256

        # test_evaluator
        metric_results = test_evaluator.evaluate(
            model=finetuning_model,
            mapped_triples=dataset.testing.mapped_triples,
            additional_filter_triples=[dataset.training.mapped_triples, dataset.validation.mapped_triples],
            use_tqdm=True,
            batch_size=256,
        )

        # log final results
        if enable_wandb:
            tracker.log_metrics(
                metrics=metric_results.to_flat_dict(),
                step=num_epochs + 1,
                prefix='test',
            )

        print("Test results")
        print(metric_results)

    else:
        return 0


if __name__ == '__main__':
    main()
