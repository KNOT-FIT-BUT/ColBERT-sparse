import os
import time
import torch
import random
import torch.nn as nn
import numpy as np
import wandb

from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints


def regularization(sparsity_scores, lmbd, reg_type: str, l0_eps=1e-6):
    if reg_type == "l0":  # Not yet used
        return lmbd * (torch.abs(sparsity_scores) > l0_eps).float().sum() / sparsity_scores.size(0)
    elif reg_type == "l1":
        return lmbd * torch.abs(sparsity_scores).sum() / sparsity_scores.size(0)
    else:
        raise NotImplementedError


def train(config: ColBERTConfig, triples, queries=None, collection=None):
    lmbd = config.lmbd
    # lmbd is the lambda hyperparameter for the sparsity scores of the loss function
    # lmbd=0.0 means no sparsity scores are used

    config.checkpoint = config.checkpoint or "bert-base-uncased"

    if config.rank < 1:
        config.help()

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print(
        "Using config.bsize =",
        config.bsize,
        "(per process) and config.accumsteps =",
        config.accumsteps,
    )

    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(
                config,
                triples,
                queries,
                collection,
                (0 if config.rank == -1 else config.rank),
                config.nranks,
            )
        else:
            reader = LazyBatcher(
                config,
                triples,
                queries,
                collection,
                (0 if config.rank == -1 else config.rank),
                config.nranks,
            )
    else:
        raise NotImplementedError()

    if not config.reranker:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    colbert = colbert.to(DEVICE)
    colbert.train()

    colbert = torch.nn.parallel.DistributedDataParallel(
        colbert,
        device_ids=[config.rank],
        output_device=config.rank,
        find_unused_parameters=True,
    )

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup,
            num_training_steps=config.maxsteps,
        )

    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999

    start_batch_idx = 0

    if config.resume:
        print("Resuming checkpoint")
        assert config.checkpoint is not None
        start_batch_idx = config.resume_batch_idx

        reader.skip_to_batch(start_batch_idx, config.bsize)

    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        print(f"Training batch [{batch_idx}/{config.maxsteps}]")
        if (warmup_bert is not None) and warmup_bert <= batch_idx:
            set_bert_grad(colbert, True)
            warmup_bert = None

        this_batch_loss = 0.0

        ## STATISTICS

        sparsity_stats_outputed = False

        ## END STATISTICS

        for batch in BatchSteps:
            with amp.context():
                try:
                    queries, passages, target_scores = batch
                    encoding = [queries, passages]
                except:
                    encoding, target_scores = batch
                    encoding = [encoding.to(DEVICE)]

                if config.use_ib_negatives:
                    scores, ib_loss, sparsity_scores = colbert(*encoding)
                else:
                    scores, sparsity_scores = colbert(*encoding)

                scores_out = scores
                doc_len = sparsity_scores.size(1)

                scores = scores.view(-1, config.nway)

                if len(target_scores) and not config.ignore_scores:
                    target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                    target_scores = target_scores * config.distillation_alpha
                    target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

                    log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                    loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(log_scores, target_scores)
                    no_sparse_loss = float(loss)
                    loss += regularization(sparsity_scores, lmbd, config.regularization)

                else:
                    loss = nn.CrossEntropyLoss()(scores, labels[: scores.size(0)])
                    no_sparse_loss = float(loss)
                    loss += regularization(sparsity_scores, lmbd, config.regularization)

                # START STATS
                if batch_idx % 10 == 0 and not sparsity_stats_outputed:
                    sparsity_stats_outputed = True
                    print("Saving stats...")

                    # SPARSITY LOSS
                    sparsity_loss = regularization(sparsity_scores, lmbd, config.regularization)

                    # SPARSITY SCORES

                    sparsity_scores_list = sparsity_scores.view(-1).tolist()
                    config.wandb_run.log(
                        {
                            "no-sparse-loss": no_sparse_loss,
                            "sparsity-loss": sparsity_loss / (lmbd if lmbd != 0.0 else 1.0),
                            "overall-loss": loss.item(),
                        }
                    )
                    config.wandb_run.log({"sparsity_scores_hist": wandb.Histogram(sparsity_scores_list, num_bins=100)})
                    config.wandb_run.log(
                        {
                            "sparsity_scores_q1": np.percentile(sparsity_scores_list, 25),
                            "sparsity_scores_q2": np.percentile(sparsity_scores_list, 50),
                            "sparsity_scores_q3": np.percentile(sparsity_scores_list, 75),
                            "sparsity_scores_q4": np.percentile(sparsity_scores_list, 100),
                        }
                    )
                # END STATS

                if config.use_ib_negatives:
                    if config.rank < 1:
                        # print('\t\t\t\t', loss.item(), ib_loss.item())
                        pass
                    loss += ib_loss

                loss = loss / config.accumsteps

            if config.rank < 1:
                pass
                # print_progress(scores)

            amp.backward(loss)

            this_batch_loss += loss.item()

        train_loss = this_batch_loss if train_loss is None else train_loss
        train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss

        amp.step(colbert, optimizer, scheduler)

        if config.rank < 1:
            print_message(batch_idx, train_loss)
            manage_checkpoints(config, colbert, optimizer, batch_idx + 1, savepath=None)

    if config.rank < 1:
        print_message("#> Done with all triples!")
        ckpt_path = manage_checkpoints(
            config,
            colbert,
            optimizer,
            batch_idx + 1,
            savepath=None,
            consumed_all_triples=True,
        )

        return ckpt_path  # TODO: This should validate and return the best checkpoint, not just the last one.


@torch.no_grad()
def validate(config: ColBERTConfig, triples, queries=None, collection=None):
    print("Validate function called")

    config.checkpoint = config.checkpoint or "bert-base-uncased"

    if config.rank < 1:
        config.help()

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print(
        "Using config.bsize =",
        config.bsize,
        "(per process) and config.accumsteps =",
        config.accumsteps,
    )

    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(
                config,
                triples,
                queries,
                collection,
                (0 if config.rank == -1 else config.rank),
                config.nranks,
            )
        else:
            reader = LazyBatcher(
                config,
                triples,
                queries,
                collection,
                (0 if config.rank == -1 else config.rank),
                config.nranks,
            )
    else:
        raise NotImplementedError()

    if not config.reranker:
        colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    colbert = colbert.to(DEVICE)
    colbert.eval()

    colbert = torch.nn.parallel.DistributedDataParallel(
        colbert,
        device_ids=[config.rank],
        output_device=config.rank,
        find_unused_parameters=True,
    )

    amp = MixedPrecisionManager(config.amp)
    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999

    start_batch_idx = 0

    # if config.resume:
    #     assert config.checkpoint is not None
    #     start_batch_idx = checkpoint['batch']

    #     reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    validation_losses = torch.Tensor([])

    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        print(f"Evaluating... {batch_idx}/{config.maxstept}", end="\r")
        if (warmup_bert is not None) and warmup_bert <= batch_idx:
            set_bert_grad(colbert, True)
            warmup_bert = None

        this_batch_loss = 0.0

        for batch in BatchSteps:
            with amp.context():
                try:
                    queries, passages, target_scores = batch
                    encoding = [queries, passages]
                except:
                    encoding, target_scores = batch
                    encoding = [encoding.to(DEVICE)]

                scores = colbert(*encoding)

                if config.use_ib_negatives:
                    scores, ib_loss = scores

                scores = scores.view(-1, config.nway)

                if len(target_scores) and not config.ignore_scores:
                    target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                    target_scores = target_scores * config.distillation_alpha
                    target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

                    log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                    loss = torch.nn.KLDivLoss(reduction="batchmean", log_target=True)(log_scores, target_scores)
                else:
                    loss = nn.CrossEntropyLoss()(scores, labels[: scores.size(0)])

                if config.use_ib_negatives:
                    if config.rank < 1:
                        print("\t\t\t\t", loss.item(), ib_loss.item())

                    loss += ib_loss

                loss = loss / config.accumsteps

            if config.rank < 1:
                print_progress(scores)

            this_batch_loss += loss.item()
            validation_losses = torch.cat((validation_losses, loss))

    torch.save(
        validation_losses,
        "/mnt/minerva1/nlp-2/homes/xsteti05/project/src/expers/colbert_sparse/EXP_VALIDATION_LOSS.pt",
    )


def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)
