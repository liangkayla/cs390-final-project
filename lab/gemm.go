package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
	"sync"
	"time"

	"gonum.org/v1/gonum/mat" // optimized matrix operations in Go
)

const Bandwidth = 50e9 // 50 GB/s going by NVLink

type Metrics struct {
	FLOPs         float64
	ComputeTime   float64
	TotalCommTime float64
	TotalTime     float64
}

type CommunicationEvent struct {
	Op          string
	NumGPUs     int
	BytesPerGPU int
	CommTime    float64
}

// GEMM represents one matrix-multiply layer in the MLP.
type GEMM struct {
	Name   string
	Split  SplitMethod
	NumGPU int

	// Inputs
	X *mat.Dense // design matrix
	A *mat.Dense // weights

	// Outputs
	Output     *mat.Dense
	CommEvents []CommunicationEvent
	SyncPoints int
}

func gemmFLOPs(m, n, k int) float64 {
	return 2.0 * float64(m*n*k)
}

// ─────────────────────────────────────────────
// Core types
// ─────────────────────────────────────────────

// SplitMethod describes how a weight matrix is partitioned across GPUs.
type SplitMethod int

const (
	NoSplit     SplitMethod = iota // baseline: single GPU, no partitioning
	ColumnSplit                    // split A along columns  → each GPU holds A[:,j:k]
	RowSplit                       // split A along rows     → each GPU holds A[i:j,:]
)

func (s SplitMethod) String() string {
	switch s {
	case NoSplit:
		return "NoSplit"
	case ColumnSplit:
		return "ColumnSplit"
	case RowSplit:
		return "RowSplit"
	default:
		return "Unknown"
	}
}

// ─────────────────────────────────────────────
// Matrix: using gonum/mat library to get true parallel matrix ops
// ─────────────────────────────────────────────

// return a new empty matrix
func NewMatrix(rows, cols int) *mat.Dense {
	return mat.NewDense(rows, cols, nil)
}

// return a new matrix filled with random floats
func NewRandMatrix(rows, cols int, rng *rand.Rand) *mat.Dense {
	data := make([]float64, rows*cols)
	for i := range data {
		data[i] = rng.NormFloat64() * 0.1
	}
	return mat.NewDense(rows, cols, data)
}

// Shape returns a string like "(4×8)".
func Shape(m *mat.Dense) string {
	r, c := m.Dims()
	return fmt.Sprintf("(%d×%d)", r, c)
}

// frobeniusError returns ‖a - b‖_F / ‖b‖_F (relative error).
func frobeniusError(a, b *mat.Dense) float64 {
	var diff mat.Dense
	diff.Sub(a, b)

	denom := mat.Norm(b, 2)
	if denom == 0 {
		return mat.Norm(&diff, 2)
	}
	return mat.Norm(&diff, 2) / denom
}

// ─────────────────────────────────────────────
// Matrix operations
// ─────────────────────────────────────────────

// MatMul computes C = A X B
func MatMul(A, B *mat.Dense) *mat.Dense {
	var C mat.Dense
	C.Mul(A, B)
	return &C
}

// MatAdd computes C = A + B (element-wise, in-place into A).
func MatAdd(A, B *mat.Dense) *mat.Dense {
	var C mat.Dense
	C.Add(A, B)
	return &C
}

// GeLU applies the Gaussian Error Linear Unit element-wise.
// Using the exact formula: 0.5 * x * (1 + erf(x / sqrt(2)))
func GeLU(m *mat.Dense) *mat.Dense {
	var out mat.Dense
	sqrt2 := math.Sqrt2

	out.Apply(func(i, j int, v float64) float64 {
		return 0.5 * v * (1 + math.Erf(v/sqrt2))
	}, m)

	return &out
}

// AllReduce simulates a sum all-reduce across `numGPUs` partitions.
// In a real system this would be a collective over the network;
// here we just sum the slice of matrices and broadcast the result.
// Returns the single summed matrix and records one communication event.
func AllReduce(parts []*mat.Dense) (*mat.Dense, CommunicationEvent) {
	start := time.Now()

	r, c := parts[0].Dims()
	bytes := 2 * r * c * 8

	simulated := time.Duration(float64(bytes) / Bandwidth * float64(time.Second))

	time.Sleep(simulated)

	var result mat.Dense
	result.CloneFrom(parts[0])
	for _, p := range parts[1:] {
		result.Add(&result, p)
	}

	commTime := time.Since(start).Seconds()

	return &result, CommunicationEvent{
		Op:          "AllReduce",
		NumGPUs:     len(parts),
		BytesPerGPU: bytes,
		CommTime:    commTime,
	}
}

// ─────────────────────────────────────────────
// Partitioning helpers
// ─────────────────────────────────────────────

// SplitColumns splits matrix M into `n` column-wise partitions.
// M has shape (r, c); each partition has shape (r, c/n).
// func SplitColumns(M *mat.Dense, n int) []*mat.Dense {
// 	r, c := M.Dims()
// 	if c%n != 0 {
// 		panic("cols not divisible")
// 	}

// 	w := c / n
// 	parts := make([]*mat.Dense, n)

// 	for i := 0; i < n; i++ {
// 		parts[i] = M.Slice(0, r, i*w, (i+1)*w).(*mat.Dense)
// 	}
// 	return parts
// }

func SplitColumns(M *mat.Dense, n int) []*mat.Dense {
	r, c := M.Dims()
	if c%n != 0 {
		panic("cols not divisible")
	}
	w := c / n
	parts := make([]*mat.Dense, n)
	for i := 0; i < n; i++ {
		part := mat.NewDense(r, w, nil)
		part.CloneFrom(M.Slice(0, r, i*w, (i+1)*w)) // ← copy, not view
		parts[i] = part
	}
	return parts
}

// SplitRows splits matrix M into `n` row-wise partitions.
// M has shape (r, c); each partition has shape (r/n, c).
// func SplitRows(M *mat.Dense, n int) []*mat.Dense {
// 	r, c := M.Dims()
// 	if r%n != 0 {
// 		panic("rows not divisible")
// 	}

// 	h := r / n
// 	parts := make([]*mat.Dense, n)

//		for i := 0; i < n; i++ {
//			parts[i] = M.Slice(i*h, (i+1)*h, 0, c).(*mat.Dense)
//		}
//		return parts
//	}
func SplitRows(M *mat.Dense, n int) []*mat.Dense {
	r, c := M.Dims()
	if r%n != 0 {
		panic("rows not divisible")
	}
	h := r / n
	parts := make([]*mat.Dense, n)
	for i := 0; i < n; i++ {
		part := mat.NewDense(h, c, nil)
		part.CloneFrom(M.Slice(i*h, (i+1)*h, 0, c)) // ← copy, not view
		parts[i] = part
	}
	return parts
}

// ConcatColumns concatenates matrices horizontally: [M1 | M2 | ...] → (r, sum(c_i))
func ConcatColumns(parts []*mat.Dense) *mat.Dense {
	r, _ := parts[0].Dims()

	totalCols := 0
	for _, p := range parts {
		_, c := p.Dims()
		totalCols += c
	}

	out := mat.NewDense(r, totalCols, nil)

	col := 0
	for _, p := range parts {
		_, c := p.Dims()
		out.Slice(0, r, col, col+c).(*mat.Dense).Copy(p)
		col += c
	}
	return out
}

// ConcatRows concatenates matrices vertically.
func ConcatRows(parts []*mat.Dense) *mat.Dense {
	_, c := parts[0].Dims()

	totalRows := 0
	for _, p := range parts {
		r, _ := p.Dims()
		totalRows += r
	}

	out := mat.NewDense(totalRows, c, nil)

	row := 0
	for _, p := range parts {
		r, _ := p.Dims()
		out.Slice(row, row+r, 0, c).(*mat.Dense).Copy(p)
		row += r
	}
	return out
}

func (e CommunicationEvent) String() string {
	return fmt.Sprintf("%s(gpus=%d, bytes/gpu=%d)", e.Op, e.NumGPUs, e.BytesPerGPU)
}

func NewGEMM(name string, split SplitMethod, numGPU int) *GEMM {
	return &GEMM{Name: name, Split: split, NumGPU: numGPU}
}

// ─────────────────────────────────────────────
// MLP Strategy implementations
// ─────────────────────────────────────────────

// MLPResult captures all metrics for one end-to-end MLP forward pass.
type MLPResult struct {
	StrategyName string
	Description  string

	Output *mat.Dense

	CommEvents     []CommunicationEvent
	SyncPoints     int
	TotalCommBytes int

	Metrics Metrics

	RelativeError float64
	Correct       bool
}

func (r *MLPResult) TotalCommMB() float64 {
	return float64(r.TotalCommBytes) / (1024 * 1024)
}

// groundTruth computes the reference output on a single GPU: GeLU(X·A1) · B
func groundTruth(X, A, B *mat.Dense) *mat.Dense {
	Y := GeLU(MatMul(X, A))
	return MatMul(Y, B)
}

// ── Task 1: strategyPaperOptimal ─────────────────────────────────────────────
//
// Column-split strategy — Figure 3(b) of Megatron-LM (Shoeybi et al., 2019).
//
// Partition:
//   - A split column-wise: GPU i holds A_i of shape (k, h/n)
//   - B split row-wise:    GPU i holds B_i of shape (h/n, c)
//   - X replicated on every GPU
//
// Forward pass on GPU i (all GPUs run in parallel):
//  1. Y_i = GeLU(X · A_i)    shape (m, h/n) — no sync needed before GeLU
//  2. Z_i = Y_i · B_i        shape (m, c)   — partial output
//     After: Z = AllReduce(Z_0, …, Z_{n-1})    — single all-reduce
//
// Key insight: column-sharding A means X·A_i is self-contained per GPU, so GeLU can be applied locally with no prior synchronisation.
func strategyPaperOptimal(X, A, B *mat.Dense, numGPU int) MLPResult {
	events := []CommunicationEvent{}
	computeTimes := make([]float64, numGPU)
	syncPts := 1 // one AllReduce on the forward pass (f in the paper)

	// TODO (1/3): Split A column-wise and B row-wise into numGPU shards.
	// Hint: use SplitColumns and SplitRows.
	var As, Bs []*mat.Dense
	_, _ = As, Bs // remove this line once you assign As and Bs

	Yparts := make([]*mat.Dense, numGPU)
	Zparts := make([]*mat.Dense, numGPU)

	// TODO (2/3): In parallel, compute Y_i = GeLU(X · A_i) and Z_i = Y_i · B_i.
	// Use a sync.WaitGroup and one goroutine per GPU (index i).
	// Record wall-clock compute time for each GPU in computeTimes[i].
	// Store results in Yparts[i] and Zparts[i].
	var wg sync.WaitGroup
	_ = wg // remove this line once you use wg

	// TODO (3/3): All-reduce the Zparts shards into the final output Z.
	// Append the returned CommunicationEvent to events.
	var Z *mat.Dense
	_ = Z // remove this line once you assign Z

	totalBytes := 0
	totalCommTime := 0.0
	for _, e := range events {
		totalBytes += e.BytesPerGPU * e.NumGPUs
		totalCommTime += e.CommTime
	}

	r, k := X.Dims()
	_, h := A.Dims()
	_, c := B.Dims()

	flopsGEMM1 := gemmFLOPs(r, h/numGPU, k) * float64(numGPU)
	flopsGEMM2 := gemmFLOPs(r, c, h/numGPU) * float64(numGPU)
	totalFLOPs := flopsGEMM1 + flopsGEMM2

	maxComputeTime := 0.0
	for _, t := range computeTimes {
		if t > maxComputeTime {
			maxComputeTime = t
		}
	}

	return MLPResult{
		StrategyName:   "Paper Optimal (col-split A, row-split B)",
		Description:    "Parallel GEMMs + 1 AllReduce",
		Output:         Z,
		CommEvents:     events,
		SyncPoints:     syncPts,
		TotalCommBytes: totalBytes,
		Metrics: Metrics{
			FLOPs:         totalFLOPs,
			ComputeTime:   maxComputeTime,
			TotalCommTime: totalCommTime,
			TotalTime:     maxComputeTime + totalCommTime,
		},
	}
}

// ── Task 2: strategyPaperSuboptimal ──────────────────────────────────────────
//
// Row-split strategy — discussed and dismissed in Megatron-LM because it requires an extra AllReduce mid-block, breaking pipeline efficiency.
//
// Partition:
//   - X split column-wise: GPU i holds X_i of shape (m, k/n)
//   - A split row-wise:    GPU i holds A_i of shape (k/n, h)
//   - B split column-wise: GPU i holds B_i of shape (h, c/n)
//
// Forward pass:
//  1. (Parallel) P_i = X_i · A_i          shape (m, h) — partial inner product
//  2. (Sync)     P = AllReduce(P_0,…,P_{n-1})           — must sum before GeLU
//  3. Y = GeLU(P)                             — applied to full sum
//  4. (Parallel) Z_i = Y · B_i            shape (m, c/n)
//  5. Z = ConcatColumns(Z_0,…) shape (m, c)  — no extra AllReduce
//
// Key insight: row-sharding A means partial products are additive fragments of XA. Since GeLU(P_0+P_1) ≠ GeLU(P_0)+GeLU(P_1), GPUs must sync before GeLU.
func strategyPaperSuboptimal(X, A, B *mat.Dense, numGPU int) MLPResult {
	events := []CommunicationEvent{}
	computeTimes := make([]float64, numGPU)
	syncPts := 1 // backward-pass conjugate (function f in the paper)

	// TODO (1/4): Split X column-wise, A row-wise, and B column-wise into numGPU shards.
	// Hint: use SplitColumns for X and B, SplitRows for A.
	var As, Bs, Xs []*mat.Dense
	_, _, _ = As, Bs, Xs // remove this line once you assign the slices

	Pparts := make([]*mat.Dense, numGPU)
	Zparts := make([]*mat.Dense, numGPU)

	// Stage 1: compute partial products P_i = X_i · A_i in parallel.
	var wg sync.WaitGroup
	for i := 0; i < numGPU; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			start := time.Now()
			// TODO (2/4): Compute P_i = MatMul(Xs[i], As[i]) and store in Pparts[i].
			computeTimes[i] = time.Since(start).Seconds()
		}(i)
	}
	wg.Wait()

	// Stage 2: AllReduce partial products, then apply GeLU.
	// This is the unavoidable mid-block sync that makes this strategy suboptimal.
	// TODO (3/4): AllReduce Pparts → Pfull, append the event to events, increment syncPts, then compute geluY = GeLU(Pfull).
	var geluY *mat.Dense
	_ = geluY // remove this line once you assign geluY

	// Stage 3: compute Z_i = geluY · B_i in parallel.
	for i := 0; i < numGPU; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			start := time.Now()
			// TODO (4/4): Compute Z_i = MatMul(geluY, Bs[i]) and store in Zparts[i].
			computeTimes[i] += time.Since(start).Seconds()
		}(i)
	}
	wg.Wait()

	// Concatenate column shards horizontally → final output Z (no AllReduce needed).
	Z := ConcatColumns(Zparts)

	totalBytes := 0
	totalCommTime := 0.0
	for _, e := range events {
		totalBytes += e.BytesPerGPU * e.NumGPUs
		totalCommTime += e.CommTime
	}

	r, k := X.Dims()
	_, h := A.Dims()
	_, c := B.Dims()

	flopsGEMM1 := gemmFLOPs(r, h, k/numGPU) * float64(numGPU)
	flopsGEMM2 := gemmFLOPs(r, c/numGPU, h) * float64(numGPU)
	totalFLOPs := flopsGEMM1 + flopsGEMM2

	maxComputeTime := 0.0
	for _, t := range computeTimes {
		if t > maxComputeTime {
			maxComputeTime = t
		}
	}

	return MLPResult{
		StrategyName:   "Paper Suboptimal (row-split A, col-split B)",
		Description:    "Parallel GEMMs + AllReduce before GeLU + parallel GEMMs, output col-concatenated",
		Output:         Z,
		CommEvents:     events,
		SyncPoints:     syncPts,
		TotalCommBytes: totalBytes,
		Metrics: Metrics{
			FLOPs:         totalFLOPs,
			ComputeTime:   maxComputeTime,
			TotalCommTime: totalCommTime,
			TotalTime:     maxComputeTime + totalCommTime,
		},
	}
}

// ── No split (single GPU baseline) ────────────────────────────────
func strategyNoSplit(X, A, B *mat.Dense, numGPU int) MLPResult {
	start := time.Now()

	Y := GeLU(MatMul(X, A))
	Z := MatMul(Y, B)

	computeTime := time.Since(start).Seconds()

	// FLOPs (full, no partition)
	r, k := X.Dims()
	_, h := A.Dims()
	_, c := B.Dims()

	flops := gemmFLOPs(r, h, k) + gemmFLOPs(r, c, h)

	metrics := Metrics{
		FLOPs:         flops,
		ComputeTime:   computeTime,
		TotalCommTime: 0,
		TotalTime:     computeTime,
	}

	return MLPResult{
		StrategyName:   "No Split (single GPU baseline)",
		Description:    "Full computation on one GPU, no communication",
		Output:         Z,
		CommEvents:     nil,
		SyncPoints:     0,
		TotalCommBytes: 0,
		Metrics:        metrics,
	}
}

// ─────────────────────────────────────────────
// Experiment runner & reporting
// ─────────────────────────────────────────────

// ExperimentConfig holds the dimensions for the experiment.
type ExperimentConfig struct {
	BatchSize int // rows of X
	DIn       int // cols of X / rows of A
	DHidden   int // cols of A / rows of B  (hidden dimension after GEMM1)
	DOut      int // cols of B
	NumGPU    int
	Seed      int64
}

func (c ExperimentConfig) String() string {
	return fmt.Sprintf(
		"batch=%d  d_in=%d  d_hidden=%d  d_out=%d  gpus=%d",
		c.BatchSize, c.DIn, c.DHidden, c.DOut, c.NumGPU,
	)
}

func runExperiment(cfg ExperimentConfig) {
	rng := rand.New(rand.NewSource(cfg.Seed))

	X := NewRandMatrix(cfg.BatchSize, cfg.DIn, rng)
	A := NewRandMatrix(cfg.DIn, cfg.DHidden, rng)
	B := NewRandMatrix(cfg.DHidden, cfg.DOut, rng)

	ref := groundTruth(X, A, B)

	strategies := []MLPResult{
		strategyNoSplit(X, A, B, 1),
		strategyPaperOptimal(X, A, B, cfg.NumGPU),
		strategyPaperSuboptimal(X, A, B, cfg.NumGPU),
	}

	const errThreshold = 1e-9

	for i := range strategies {
		s := &strategies[i]
		s.RelativeError = frobeniusError(s.Output, ref)
		s.Correct = s.RelativeError < errThreshold
	}

	printReport(cfg, strategies)
}

func printReport(cfg ExperimentConfig, results []MLPResult) {
	sep := strings.Repeat("─", 100)
	fmt.Println()
	fmt.Println(strings.Repeat("═", 100))
	fmt.Println("  MEGATRON-LM MLP SPLIT STRATEGY EXPERIMENT")
	fmt.Println("  " + cfg.String())
	fmt.Println(strings.Repeat("═", 100))

	for i, r := range results {
		fmt.Printf("\n[%d] %s\n", i+1, r.StrategyName)
		fmt.Printf("    %s\n", r.Description)
		fmt.Println(sep)

		// Communication events
		if len(r.CommEvents) == 0 {
			fmt.Println("    Communication : none")
		} else {
			for j, e := range r.CommEvents {
				fmt.Printf("    Comm event #%d : %s\n", j+1, e)
			}
		}

		// Metrics table
		fmt.Printf("    FLOPs            : %.3e\n", r.Metrics.FLOPs)
		fmt.Printf("    Compute time     : %.6f ms\n", r.Metrics.ComputeTime*math.Pow10(3))
		fmt.Printf("    Comm time        : %.6f ms\n", r.Metrics.TotalCommTime*math.Pow10(3))
		fmt.Printf("    Total time       : %.6f ms\n", r.Metrics.TotalTime*math.Pow10(3))
		fmt.Printf("    Sync points      : %d\n", r.SyncPoints)
		fmt.Printf("    Total comm data  : %.3f MB\n", r.TotalCommMB())
		fmt.Printf("    Relative error   : %.2e  ", r.RelativeError)

		if r.Correct {
			fmt.Println("✓ CORRECT")
		} else {
			fmt.Println("✗ NUMERICALLY WRONG (non-linear barrier missing!)")
		}
	}

}

// ─────────────────────────────────────────────
//  main — tunable experiments
// ─────────────────────────────────────────────

func main() {
	fmt.Print("Experiment A: small model \n")
	// ── Experiment A: small, easy to follow ──────────────────────────────────
	runExperiment(ExperimentConfig{
		BatchSize: 4,
		DIn:       8,
		DHidden:   16,
		DOut:      8,
		NumGPU:    2,
		Seed:      42,
	})

	fmt.Print("Experiment B: medium model, more GPUS helps \n")
	// ── Experiment B: larger hidden dim, 4 GPUs ───────────────────────────────
	runExperiment(ExperimentConfig{
		BatchSize: 8,
		DIn:       64,
		DHidden:   256,
		DOut:      64,
		NumGPU:    4,
		Seed:      7,
	})

	fmt.Print("Experiment C: large model, more GPUS should help \n")
	// ── Experiment C: near transformer scale, 8 GPUs ─────────────────────────
	runExperiment(ExperimentConfig{
		BatchSize: 8,
		DIn:       512,
		DHidden:   2048,
		DOut:      512,
		NumGPU:    8,
		Seed:      99,
	})

	fmt.Print("Experiment D: large model with 16 GPUs \n")
	// ── Experiment D: near transformer scale, 16 GPUs ─────────────────────────
	runExperiment(ExperimentConfig{
		BatchSize: 24,
		DIn:       512,
		DHidden:   2048,
		DOut:      512,
		NumGPU:    16,
		Seed:      99,
	})

	fmt.Print("Experiment E: large model with 24 GPUs \n")
	// ── Experiment E: near transformer scale, 24 GPUs ─────────────────────────
	runExperiment(ExperimentConfig{
		BatchSize: 24,
		DIn:       512,
		DHidden:   2048,
		DOut:      512,
		NumGPU:    32,
		Seed:      99,
	})
}
