# Data pipeline

Files: `src/data/dataset.py`, `src/data/dataloader.py`, `src/data/tokenizer.py`;
wiring in `src/training/trainer.py`.

## Datasets

### `SyntheticTokenDataset`

A learnable synthetic language: a fixed random transition table
`next_tokens[vocab, k=4]` (seeded) and sequences produced by walking it with a
per-sample generator `seed·1_000_003 + idx`. Each sample is a pure function of its
index, so it is deterministic and independent of worker count, rank or epoch. The
entropy floor is `ln 4 ≈ 1.386` nats/token, so a model that learns the table drives
the loss from `ln(V)` towards ~1.39 — which is what the convergence benchmark shows.
It is generated in a Python loop (`seq_len` `torch.randint` calls per item), so for
large `seq_len` it is CPU-bound; this is irrelevant for the tiny runs here but not
a realistic input pipeline.

Default sizes (`Trainer._default_dataset`): 4096 training and 256 validation
samples, so with `micro_batch_size=4`, `dp=2` an epoch is 512 micro-batches.

### `PackedTokenDataset`

A memory-mapped flat `uint16` file; window `i` is tokens `[i·S, i·S + S + 1)`,
giving `(n_tokens - 1) // S` windows with inputs/labels shifted by one. `uint16`
caps the vocabulary at 65,536 ids (fine for GPT-2's 50,257; **not** for 128k-vocab
tokenisers). The memmap is opened lazily per worker process. **There is no
script in the repository that produces such a file**, and `data.tokenizer_name` is
not read by the trainer; `src/data/tokenizer.py` (`HFTokenizer`, `ByteTokenizer`)
is a library utility only. To train on real text you must tokenise and write the
`uint16` file yourself (for GPT-2: `tiktoken`/`transformers` `encode`, append EOS per
document, `np.array(ids, dtype=np.uint16).tofile(path)`).

### Tokenizers

`build_tokenizer("")` returns a dependency-free `ByteTokenizer` (vocab 256);
otherwise `HFTokenizer(name)` wraps `AutoTokenizer.from_pretrained` (needs
`transformers`; `gpt2` is public, `meta-llama/*` is gated). `HFTokenizer.vocab_size`
excludes added special tokens — size the model's `vocab_size` accordingly (the
125m config rounds 50,257 up to 50,304).

## Sharding across data-parallel ranks

`ShardedSampler(dataset_len, num_replicas=dp_size, rank=dp_rank, shuffle, seed)`:

1. Build the index list: a `torch.randperm` seeded with `seed + epoch` (identical on
   every DP rank) or `range(n)` when not shuffling.
2. Truncate to `len(self)·num_replicas` where `len(self) = dataset_len // num_replicas`.
3. Take the strided slice `indices[rank::num_replicas]`.

Properties (tested in `tests/integration/test_dataloader_sharding.py`): ranks are
disjoint; every rank yields **exactly** `dataset_len // dp` samples; at most
`dp - 1` samples are dropped per epoch (a different set each epoch when shuffling);
same seed and epoch reproduces the order; different epochs reshuffle.

**Bug fixed during the audit:** the previous implementation kept the remainder, so
shard lengths differed by one. With `drop_last=True` DataLoaders that can mean
different *batch counts* per rank at an epoch boundary, so one rank starts the next
epoch (and its collectives) while another is still finishing — a hang. The
sampler now guarantees equal lengths (commit "fix(data): make ShardedSampler shards
equal-length across DP ranks").

TP ranks share a `dp_rank`, hence receive identical batches — required because the
row-parallel all-reduce sums activations that must come from the same tokens.

## Epoch handling and resume

`Trainer._batch_iterator` is an infinite generator: it calls
`sampler.set_epoch(self._epoch)` before iterating the loader and increments the epoch
when the loader is exhausted. The validation loader never shuffles and is re-iterated
from the start on every evaluation (the first `eval_steps` batches).

There is no dataloader state in a checkpoint. On resume
`Trainer._fast_forward_data(step)` recreates the iterator at epoch 0 and **replays**
`step·grad_accum_steps` batches. Because samples are pure functions of
(seed, epoch, index), the stream after replay is identical to a continuous run (the
resume benchmark shows bit-identical losses). The cost is O(step) data loading at
start-up — negligible for the synthetic set, significant for a 100k-step run on real
data (a seekable sampler state would remove it; see Limitations).

## Global batch identity

`global_batch_size = micro_batch_size · grad_accum_steps · dp_size`. A value `<= 0`
is derived at start-up; a mismatch raises with the arithmetic spelled out
(`TrainingConfig.validate`). `Trainer._assert_consistent_global_batch` additionally
all-reduces the three batch settings with `MAX` and aborts if any rank differs.
