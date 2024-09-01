from colbert.infra.config.config import ColBERTConfig
from colbert.search.strided_tensor import StridedTensor
from colbert.utils.utils import print_message, flatten
from colbert.modeling.base_colbert import BaseColBERT
from colbert.parameters import DEVICE

import torch
import string

import os
import pathlib
from torch.utils.cpp_extension import load


class ColBERT(BaseColBERT):
    """
        This class handles the basic encoding and scoring operations in ColBERT. It is used for training.
    """

    def __init__(self, name='bert-base-uncased', colbert_config=None):
        super().__init__(name, colbert_config)
        self.use_gpu = colbert_config.total_visible_gpus > 0

        ColBERT.try_load_torch_extensions(self.use_gpu)

        if self.colbert_config.mask_punctuation:
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.raw_tokenizer.encode(symbol, add_special_tokens=False)[0]]}
        self.pad_token = self.raw_tokenizer.pad_token_id
        
        # STATS 
        self.keep_stats = []
        self.discard_stats = []
        
    @classmethod
    def try_load_torch_extensions(cls, use_gpu):
        if hasattr(cls, "loaded_extensions") or use_gpu:
            return

        print_message(f"Loading segmented_maxsim_cpp extension (set COLBERT_LOAD_TORCH_EXTENSION_VERBOSE=True for more info)...")
        segmented_maxsim_cpp = load(
            name="segmented_maxsim_cpp",
            sources=[
                os.path.join(
                    pathlib.Path(__file__).parent.resolve(), "segmented_maxsim.cpp"
                ),
            ],
            extra_cflags=["-O3"],
            verbose=os.getenv("COLBERT_LOAD_TORCH_EXTENSION_VERBOSE", "False") == "True",
        )
        cls.segmented_maxsim = segmented_maxsim_cpp.segmented_maxsim_cpp

        cls.loaded_extensions = True

    def forward(self, Q, D):
        Q = self.query(*Q)
        D, D_mask, sparsity_scores = self.doc(*D, keep_dims='return_mask')

        # Repeat each query encoding for every corresponding document.
        Q_duplicated = Q.repeat_interleave(self.colbert_config.nway, dim=0).contiguous()
        scores = self.score(Q_duplicated, D, D_mask)

        if self.colbert_config.use_ib_negatives:
            ib_loss = self.compute_ib_loss(Q, D, D_mask)
            return scores, ib_loss, sparsity_scores

        return scores, sparsity_scores

    def compute_ib_loss(self, Q, D, D_mask):
        # TODO: Organize the code below! Quite messy.
        scores = (D.unsqueeze(0) @ Q.permute(0, 2, 1).unsqueeze(1)).flatten(0, 1)  # query-major unsqueeze

        scores = colbert_score_reduce(scores, D_mask.repeat(Q.size(0), 1, 1), self.colbert_config)

        nway = self.colbert_config.nway
        all_except_self_negatives = [list(range(qidx*D.size(0), qidx*D.size(0) + nway*qidx+1)) +
                                     list(range(qidx*D.size(0) + nway * (qidx+1), qidx*D.size(0) + D.size(0)))
                                     for qidx in range(Q.size(0))]

        scores = scores[flatten(all_except_self_negatives)]
        scores = scores.view(Q.size(0), -1)  # D.size(0) - self.colbert_config.nway + 1)

        labels = torch.arange(0, Q.size(0), device=scores.device) * (self.colbert_config.nway)

        return torch.nn.CrossEntropyLoss()(scores, labels)

    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        mask = torch.tensor(self.mask(input_ids, skiplist=[]), device=self.device).unsqueeze(2).float()
        Q = Q * mask

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True, include_sparsity_scores=True):
        assert keep_dims in [True, False, 'return_mask']

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        D = self.bert(input_ids, attention_mask=attention_mask)[0]
        D = self.linear(D)
        sparsity_scores = self.slinear(D)
        sparsity_scores = self.ssigmoid(sparsity_scores)
        
        mask = torch.tensor(self.mask(input_ids, skiplist=self.skiplist), device=self.device).unsqueeze(2).float()
        rhat = D * mask
        rhat = torch.nn.functional.normalize(rhat, p=2, dim=2)
        
        rhat_shape = rhat.shape
        sparsity_shape = sparsity_scores.shape
        
        rhat = rhat.view(-1, rhat.size(-1)).unsqueeze(-1)
        sparsity_scores = sparsity_scores.view(-1, sparsity_scores.size(-1)).unsqueeze(-1)
        
        r = torch.bmm(rhat, sparsity_scores)
        r = r.view(rhat_shape)
        if self.use_gpu:
            r = r.half()
            
        if self.colbert_config.sparse_reduce and keep_dims in ["return_mask", False]: # INDEXING MODE - use sparsity scores to reduce # of embs
            # 3 experiments - thesholding: abs(sparse_score) < delta param
            #               - probability cutoff: prob.distr(sparse_scores) -> leave top x-quantile
            #               - top-k: eliminate fixed number of embs
            
            sparse_reduce_type = self.colbert_config.sparse_reduce_type
            assert self.colbert_config.sparse_reduce_type in ["threshold", "prob_cutoff", "top_k"]

            sparsity_scores = sparsity_scores.view(sparsity_shape)

            # s_scores (batch_size, doclen, 1)
            # mask (batch_size, doclen, 1 )
            # r (batch_size, doclen, config.dim)

            if sparse_reduce_type == "threshold": 
                delta = self.colbert_config.sparse_reduce_delta
                keep_mask = sparsity_scores.abs() >= delta
             
                # Stats
                keep_stats = torch.sum(keep_mask, dim=1)
                discard_stats = torch.sum(~keep_mask, dim=1)
             
                self.keep_stats.append(keep_stats)  
                self.discard_stats.append(discard_stats)                
             
                mask = mask.bool() & keep_mask                
          
                
            elif sparse_reduce_type == "prob_cutoff": 
                quantile = self.colbert_config.sparse_reduce_quantile
                
                prob_distr = torch.nn.functional.softmax(sparsity_scores, dim=1)
                probs_sorted, sorted_indices = prob_distr.squeeze(-1).sort(dim=1, descending=True)
                
                cum_probs = probs_sorted.cumsum(dim=1)
                cutoff_mask = cum_probs <= quantile
                
                cutoff_indices = cutoff_mask.sum(dim=1, keep_dims= True)
                
                select_mask = torch.arange(r.size(1), device=self.device).unsqueeze(0) < cutoff_indices
                
                r = torch.gather(r, 1, sorted_indices.unsqueeze(-1).expand_as(r))[select_mask]\
                    .view(r.size(0), -1, r.size(2))
                sparsity_scores = torch.gather(sparsity_scores, 1, sorted_indices.unsqueeze(-1))[select_mask]\
                    .view(sparsity_scores.size(0), -1, 1)
                mask = torch.gather(mask, 1, sorted_indices.unsqueeze(-1).expand_as(mask))[select_mask]\
                    .view(mask.size(0), -1, 1)


            elif sparse_reduce_type == "top_k": 
                # TODO: implement 
                # k = self.colbert_config.sparse_reduce_k
                pass
            
            
        if keep_dims is False:
            r, mask = rhat.cpu(), mask.bool().cpu().squeeze(-1)
            r = [d[mask[idx]] for idx, d in enumerate(r)]

        elif keep_dims == 'return_mask':
            return r, mask.bool()

        if include_sparsity_scores:
            return r, sparsity_scores
        return r

    def score(self, Q, D_padded, D_mask):
        # assert self.colbert_config.similarity == 'cosine'
        if self.colbert_config.similarity == 'l2':
            assert self.colbert_config.interaction == 'colbert'
            return (-1.0 * ((Q.unsqueeze(2) - D_padded.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)
        return colbert_score(Q, D_padded, D_mask, config=self.colbert_config)

    def mask(self, input_ids, skiplist):
        mask = [[(x not in skiplist) and (x != self.pad_token) for x in d] for d in input_ids.cpu().tolist()]
        return mask
    
    def output_stats(self):
        if len(self.keep_stats) > 0:
            keep_stats = torch.cat(self.keep_stats, dim=0)
            discard_stats = torch.cat(self.discard_stats, dim=0)
            
            param_str = ""
            if self.colbert_config.sparse_reduce_type == "threshold":
                param_str = f"delta-{self.colbert_config.sparse_reduce_delta}"
            elif self.colbert_config.sparse_reduce_type == "prob_cutoff":
                param_str = f"quantile-{self.colbert_config.sparse_reduce_quantile}"
            elif self.colbert_config.sparse_reduce_type == "top_k":
                param_str = f"k-{self.colbert_config.sparse_reduce_k}"
            
            save_dir_path = "/home/xsteti05/mnt/karolina/projects/colbert_sparse/outputs/stats/index_stats"
            
            torch.save(keep_stats, os.path.join(save_dir_path, f"{param_str}_keep_mask_stats_keep.pt"))
            torch.save(discard_stats, os.path.join(save_dir_path, f"{param_str}_keep_mask_stats_discard.pt"))
            print("DEBUG: Saved keep_mask_stats_keep.pt and keep_mask_stats_discard.pt")
        else:
            print("DEBUG: No stats to save")
    


# TODO: In Query/DocTokenizer, use colbert.raw_tokenizer

# TODO: The masking below might also be applicable in the kNN part
def colbert_score_reduce(scores_padded, D_mask, config: ColBERTConfig):
    D_padding = ~D_mask.view(scores_padded.size(0), scores_padded.size(1)).bool()
    scores_padded[D_padding] = -9999
    scores = scores_padded.max(1).values

    assert config.interaction in ['colbert', 'flipr'], config.interaction

    if config.interaction == 'flipr':
        assert config.query_maxlen == 64, ("for now", config)
        # assert scores.size(1) == config.query_maxlen, scores.size()

        K1 = config.query_maxlen // 2
        K2 = 8

        A = scores[:, :config.query_maxlen].topk(K1, dim=-1).values.sum(-1)
        B = 0

        if K2 <= scores.size(1) - config.query_maxlen:
            B = scores[:, config.query_maxlen:].topk(K2, dim=-1).values.sum(1)

        return A + B

    return scores.sum(-1)


# TODO: Wherever this is called, pass `config=`
def colbert_score(Q, D_padded, D_mask, config=ColBERTConfig()):
    """
        Supply sizes Q = (1 | num_docs, *, dim) and D = (num_docs, *, dim).
        If Q.size(0) is 1, the matrix will be compared with all passages.
        Otherwise, each query matrix will be compared against the *aligned* passage.

        EVENTUALLY: Consider masking with -inf for the maxsim (or enforcing a ReLU).
    """

    use_gpu = config.total_visible_gpus > 0
    if use_gpu:
        Q, D_padded, D_mask = Q.cuda(), D_padded.cuda(), D_mask.cuda()

    assert Q.dim() == 3, Q.size()
    assert D_padded.dim() == 3, D_padded.size()
    assert Q.size(0) in [1, D_padded.size(0)]

    scores = D_padded @ Q.to(dtype=D_padded.dtype).permute(0, 2, 1)

    return colbert_score_reduce(scores, D_mask, config)


def colbert_score_packed(Q, D_packed, D_lengths, config=ColBERTConfig()):
    """
        Works with a single query only.
    """

    use_gpu = config.total_visible_gpus > 0

    if use_gpu:
        Q, D_packed, D_lengths = Q.cuda(), D_packed.cuda(), D_lengths.cuda()

    Q = Q.squeeze(0)

    assert Q.dim() == 2, Q.size()
    assert D_packed.dim() == 2, D_packed.size()

    scores = D_packed @ Q.to(dtype=D_packed.dtype).T

    if use_gpu or config.interaction == "flipr":
        scores_padded, scores_mask = StridedTensor(scores, D_lengths, use_gpu=use_gpu).as_padded_tensor()

        return colbert_score_reduce(scores_padded, scores_mask, config)
    else:
        return ColBERT.segmented_maxsim(scores, D_lengths)
