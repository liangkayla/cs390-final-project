# GEMM

The goal of this lab is to implement two tensor-parallel strategies for a
two-layer MLP forward pass, based on Figure 3 of the
[Megatron-LM paper](https://arxiv.org/abs/1909.08053) (Shoeybi et al., 2019).
Reading Section 3.2 of that paper is important for completing the project.

The two strategies differ in *how* the weight matrices are partitioned across
GPUs and *when* the GPUs must synchronise.  Understanding that difference is
the core lesson of this lab.

## Background

### The MLP block

The forward pass of a standard two-layer MLP with GeLU activation is:

```
Z = GeLU(X · A) · B
```

where:
- `X`  is the input design matrix of shape `(m, k)` — one row per token in the batch
- `A`  is the first weight matrix of shape `(k, h)` — projects to the hidden dimension
- `B`  is the second weight matrix of shape `(h, c)` — projects to the output dimension
- `GeLU` is the Gaussian Error Linear Unit, applied element-wise

On a single GPU the computation is sequential: GEMM → GeLU → GEMM.  The lab
asks you to distribute this across `n` GPUs and measure the resulting
communication overhead.

### Tensor parallelism

Tensor parallelism (also called model parallelism within a layer) splits
individual weight matrices across GPUs so that each GPU holds and operates on
only a shard of the parameters.  After the local computation, GPUs
communicate to assemble the full result.

The key design question is: **which dimension do you split, and where do you
put the synchronisation barrier?**  The answer determines both correctness
and efficiency.

### Communication primitive

The only collective communication you will use is **AllReduce**: each GPU
holds a partial matrix of the same shape, all partials are summed, and every
GPU receives the global sum.  The simulated cost is:

```
comm_time = (2 × bytes_per_gpu) / Bandwidth
```

where `Bandwidth = 50 GB/s` models NVLink.

### Timers and goroutines

Because this is a simulation, parallelism is modelled with Go goroutines and
`sync.WaitGroup`.  Each goroutine records its own wall-clock compute time;
the reported `ComputeTime` is the maximum across GPUs (i.e., the critical path).

## Tasks

Complete **two functions** in `gemm.go`.  Do not modify any other files.

---

### Task 1: `strategyPaperOptimal`

This is the **column-split** strategy described as optimal in the Megatron-LM
paper (Figure 3b).  It requires only a **single AllReduce** at the very end of
the forward pass.

**Partition:**
| Tensor | Split | Each GPU holds |
|--------|-------|----------------|
| `X`    | replicated | full `(m, k)` |
| `A`    | column-wise | `(k, h/n)` shard |
| `B`    | row-wise | `(h/n, c)` shard |

**Forward pass on GPU i** (all GPUs run in parallel):
1. `Y_i = GeLU(X · A_i)`   — shape `(m, h/n)`, no sync needed before GeLU
2. `Z_i = Y_i · B_i`       — shape `(m, c)`, partial output

**After parallel stages:**
```
Z = AllReduce(Z_0, Z_1, …, Z_{n-1})
```

Because each GPU holds a *disjoint column shard*
of `A`, the product `X·A_i` is already self-contained — it does not need to
be summed with any other GPU's result before the activation.  GeLU can be
applied locally, with no synchronisation.

**Expected properties of your implementation:**
- `SyncPoints == 1`
- `len(CommEvents) == 1`, with `CommEvents[0].Op == "AllReduce"`
- Output matches `groundTruth` to within `1e-9` relative error

---

### Task 2: `strategyPaperSuboptimal`

This is the **row-split** strategy discussed in the paper, which is suboptimal
because it requires an **extra AllReduce** in the middle of the block, breaking
pipeline efficiency.

**Partition:**
| Tensor | Split | Each GPU holds |
|--------|-------|----------------|
| `X`    | column-wise | `(m, k/n)` shard |
| `A`    | row-wise | `(k/n, h)` shard |
| `B`    | column-wise | `(h, c/n)` shard |

**Forward pass:**
1. *(Parallel)* `P_i = X_i · A_i`   — shape `(m, h)`, partial inner product
2. *(Sync)*     `P = AllReduce(P_0, …, P_{n-1})` — must sum before activation
3. `Y = GeLU(P)` — applied once to the full sum
4. *(Parallel)* `Z_i = Y · B_i`    — shape `(m, c/n)`, column shard of output
5. `Z = ConcatColumns(Z_0, …, Z_{n-1})` — assemble final output, no AllReduce needed

Because each GPU holds a disjoint *row shard* of `A` (the inner dimension), the partial products `X_i·A_i` are
additive fragments of the full `XA`.  Since GeLU is non-linear:

```
GeLU(P_0 + P_1) ≠ GeLU(P_0) + GeLU(P_1)
```

The GPUs *must* synchronise and sum their partials before any GPU can apply
GeLU.  This mid-block AllReduce is the cost that makes this strategy worse than the optimal strategy.

**Expected properties of your implementation:**
- `SyncPoints == 2`
- `len(CommEvents) == 1`, with `CommEvents[0].Op == "AllReduce"`
- Output matches `groundTruth` to within `1e-9` relative error

---

## Methods to Implement

Fill in the implementation for these two functions in `gemm.go`:

```go
func strategyPaperOptimal(X, A, B *mat.Dense, numGPU int) MLPResult
func strategyPaperSuboptimal(X, A, B *mat.Dense, numGPU int) MLPResult
```

The detailed algorithms are spelled out in the docstrings above each function.
Study `strategyNoSplit` as a reference.

### Helper methods (implemented)

Many useful helper methods have already been implemented in the starter code. They have been defined below.

| Function | Description |
|----------|-------------|
| `SplitColumns(M, n)` | Split `M` into `n` column shards |
| `SplitRows(M, n)` | Split `M` into `n` row shards |
| `ConcatColumns(parts)` | Horizontal concatenation |
| `ConcatRows(parts)` | Vertical concatenation |
| `MatMul(A, B)` | Matrix multiply, returns new matrix |
| `GeLU(M)` | Element-wise GeLU, returns new matrix |
| `AllReduce(parts)` | Simulated all-reduce, returns `(*mat.Dense, CommunicationEvent)` |
| `gemmFLOPs(m, n, k)` | Returns `2·m·n·k` (multiply-add count) |
| `groundTruth(X, A, B)` | Reference single-GPU output for correctness checking |

## Running Tests

```bash
# Run all tests (should fail until you complete the implementation for the two strategy functions):
go test -v

# Run a single test:
go test -v -run TestPaperOptimal_SmallTwoGPU

# Run the full experiment (prints a formatted report for five configs):
go run gemm.go
```