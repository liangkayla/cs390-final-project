package cs390finalproject
// megatron_mlp.go
//
// Minimal Go experiment modelling Megatron-LM tensor parallelism for an MLP block.
//
// Architecture (matches diagram):
//
//   X  ──► f ──► [shard 0: X·A1 → GeLU → Y1]  ──► [shard 0: Y1·B1 → Z1] ──► g ──► Z
//               [shard 1: X·A2 → GeLU → Y2]       [shard 1: Y2·B2 → Z2]
//
//   1st GEMM  (column-parallel):  A = [A1 | A2]  – split A along columns.
//             Each shard gets the full X and produces Y_k (a column slice of Y).
//             No inter-shard communication needed; results concatenate naturally.
//             f = identity forward, all-reduce backward (not shown here).
//
//   2nd GEMM  (row-parallel):     B = [B1 ; B2]  – split B along rows.
//             Each shard owns the matching Y_k row-slice and produces a partial Z_k.
//             All-reduce across shards to sum partials → full Z.
//             g = all-reduce forward, identity backward (not shown here).
//
// Every "shard" is a goroutine simulating an independent GPU rank.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"sync"
)

// ── Matrix ────────────────────────────────────────────────────────────────────

type Matrix struct {
	rows, cols int
	data       []float64
}

func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{rows: rows, cols: cols, data: make([]float64, rows*cols)}
}

func (m *Matrix) At(r, c int) float64     { return m.data[r*m.cols+c] }
func (m *Matrix) Set(r, c int, v float64) { m.data[r*m.cols+c] = v }

func RandMatrix(rows, cols int, rng *rand.Rand) *Matrix {
	m := NewMatrix(rows, cols)
	for i := range m.data {
		m.data[i] = rng.NormFloat64() * 0.02
	}
	return m
}

// matmul: C = A (m×k) · B (k×n)
func matmul(A, B *Matrix) *Matrix {
	if A.cols != B.rows {
		panic(fmt.Sprintf("matmul: A.cols=%d != B.rows=%d", A.cols, B.rows))
	}
	C := NewMatrix(A.rows, B.cols)
	for i := 0; i < A.rows; i++ {
		for j := 0; j < B.cols; j++ {
			s := 0.0
			for k := 0; k < A.cols; k++ {
				s += A.At(i, k) * B.At(k, j)
			}
			C.Set(i, j, s)
		}
	}
	return C
}

// ── Activation ────────────────────────────────────────────────────────────────

// gelu: GeLU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
func gelu(x float64) float64 {
	return 0.5 * x * (1 + math.Tanh(math.Sqrt(2/math.Pi)*(x+0.044715*x*x*x)))
}

func applyGeLU(m *Matrix) *Matrix {
	out := NewMatrix(m.rows, m.cols)
	for i, v := range m.data {
		out.data[i] = gelu(v)
	}
	return out
}

// ── Column-split helpers ───────────────────────────────────────────────────────

// splitCols divides a matrix into numShards column-slices.
// If cols is not evenly divisible, the last shard gets the remainder.
func splitCols(m *Matrix, numShards int) []*Matrix {
	base := m.cols / numShards
	shards := make([]*Matrix, numShards)
	col := 0
	for s := 0; s < numShards; s++ {
		w := base
		if s == numShards-1 {
			w = m.cols - col // give remainder to last shard
		}
		shard := NewMatrix(m.rows, w)
		for r := 0; r < m.rows; r++ {
			for c := 0; c < w; c++ {
				shard.Set(r, c, m.At(r, col+c))
			}
		}
		shards[s] = shard
		col += w
	}
	return shards
}

// splitRows divides a matrix into numShards row-slices.
func splitRows(m *Matrix, numShards int) []*Matrix {
	base := m.rows / numShards
	shards := make([]*Matrix, numShards)
	row := 0
	for s := 0; s < numShards; s++ {
		h := base
		if s == numShards-1 {
			h = m.rows - row
		}
		shard := NewMatrix(h, m.cols)
		for r := 0; r < h; r++ {
			copy(shard.data[r*m.cols:], m.data[(row+r)*m.cols:(row+r)*m.cols+m.cols])
		}
		shards[s] = shard
		row += h
	}
	return shards
}

// allReduceSum sums a slice of same-shape matrices element-wise (simulates NCCL all-reduce).
func allReduceSum(parts []*Matrix) *Matrix {
	out := NewMatrix(parts[0].rows, parts[0].cols)
	for _, p := range parts {
		for i, v := range p.data {
			out.data[i] += v
		}
	}
	return out
}

// ── Megatron tensor-parallel MLP ──────────────────────────────────────────────

// MegatronMLP runs one MLP forward pass using Megatron-style tensor parallelism.
//
//   X  (seqLen × hiddenDim)
//   A  (hiddenDim × ffnDim)   – 1st weight, column-parallel
//   B  (ffnDim × hiddenDim)   – 2nd weight, row-parallel
//
// Returns Z (seqLen × hiddenDim).
func MegatronMLP(X, A, B *Matrix, numShards int) *Matrix {
	// ── Partition weights ──────────────────────────────────────────────────────
	//
	// Column-split A  → A_k has shape (hiddenDim × ffnDim/P)
	// Row-split    B  → B_k has shape (ffnDim/P  × hiddenDim)
	//
	// Matching splits: shard k owns A_k columns and B_k rows, so
	// Y_k = GeLU(X · A_k)  and  Z_k = Y_k · B_k  are self-contained.

	Ashards := splitCols(A, numShards) // column-parallel  (1st GEMM)
	Bshards := splitRows(B, numShards) // row-parallel     (2nd GEMM)

	partialZ := make([]*Matrix, numShards)
	var wg sync.WaitGroup

	for s := 0; s < numShards; s++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			// f  = identity (forward).  X is broadcast to every shard.
			// 1st GEMM: column-parallel – no all-reduce needed.
			Yk := applyGeLU(matmul(X, Ashards[id])) // (seqLen × ffnDim/P)

			// 2nd GEMM: row-parallel – produces a partial sum.
			partialZ[id] = matmul(Yk, Bshards[id]) // (seqLen × hiddenDim)

			fmt.Printf("  [shard %d] 1st GEMM out: %dx%d  |  2nd GEMM partial: %dx%d\n",
				id, Yk.rows, Yk.cols, partialZ[id].rows, partialZ[id].cols)
		}(s)
	}
	wg.Wait()

	// g = all-reduce (forward): sum partial Zs across shards.
	Z := allReduceSum(partialZ)
	return Z
}

// ── Baseline (single-device, for correctness check) ───────────────────────────

func baselineMLP(X, A, B *Matrix) *Matrix {
	return matmul(applyGeLU(matmul(X, A)), B)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// maxAbsDiff returns the largest |a-b| across all elements (for sanity check).
func maxAbsDiff(a, b *Matrix) float64 {
	maxD := 0.0
	for i := range a.data {
		if d := math.Abs(a.data[i] - b.data[i]); d > maxD {
			maxD = d
		}
	}
	return maxD
}

// ── Main ──────────────────────────────────────────────────────────────────────

func main() {
	rng := rand.New(rand.NewSource(42))

	// Dimensions (keep small for readability; scale up to stress-test).
	seqLen    := 4   // tokens in the batch
	hiddenDim := 8   // model width
	ffnDim    := 16  // FFN intermediate width (typically 4× hidden)
	numShards := 2   // simulated GPU count (must divide ffnDim evenly)

	fmt.Printf("=== Megatron Tensor-Parallel MLP ===\n")
	fmt.Printf("seqLen=%d  hiddenDim=%d  ffnDim=%d  shards=%d\n\n",
		seqLen, hiddenDim, ffnDim, numShards)

	X := RandMatrix(seqLen, hiddenDim, rng) // input
	A := RandMatrix(hiddenDim, ffnDim, rng) // 1st weight (column-parallel)
	B := RandMatrix(ffnDim, hiddenDim, rng) // 2nd weight (row-parallel)

	fmt.Println("--- Parallel forward pass ---")
	Zpar := MegatronMLP(X, A, B, numShards)

	fmt.Println("\n--- Baseline (single device) ---")
	Zbase := baselineMLP(X, A, B)

	diff := maxAbsDiff(Zpar, Zbase)
	fmt.Printf("\nmax |Z_parallel - Z_baseline| = %.2e  ", diff)
	if diff < 1e-9 {
		fmt.Println("✓ outputs match")
	} else {
		fmt.Println("✗ outputs differ – check shard logic")
	}

	fmt.Println("\nZ (first row):")
	for c := 0; c < Zpar.cols; c++ {
		fmt.Printf("  [%d] parallel=%.4f  baseline=%.4f\n", c, Zpar.At(0, c), Zbase.At(0, c))
	}
}