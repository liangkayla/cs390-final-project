package cs390finalproject

import (
	"fmt"
	"math"
	"math/rand"
	"strings"
)

// ─────────────────────────────────────────────
// SECTION 1: Core types
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
// SECTION 2: Matrix
// ─────────────────────────────────────────────

// Matrix is a row-major dense matrix.
type Matrix struct {
	Rows, Cols int
	Data       []float64
}

// create an empty new matrix
func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{Rows: rows, Cols: cols, Data: make([]float64, rows*cols)}
}

// create a new matrix filled with random floats
func NewRandMatrix(rows, cols int, rng *rand.Rand) *Matrix {
	m := NewMatrix(rows, cols)
	for i := range m.Data {
		m.Data[i] = rng.NormFloat64() * 0.1
	}
	return m
}

func (m *Matrix) At(r, c int) float64     { return m.Data[r*m.Cols+c] }
func (m *Matrix) Set(r, c int, v float64) { m.Data[r*m.Cols+c] = v }

// Clone returns a deep copy.
func (m *Matrix) Clone() *Matrix {
	n := NewMatrix(m.Rows, m.Cols)
	copy(n.Data, m.Data)
	return n
}

// Shape returns a string like "(4×8)".
func (m *Matrix) Shape() string {
	return fmt.Sprintf("(%d×%d)", m.Rows, m.Cols)
}

// frobeniusNorm returns ‖m‖_F.
func frobeniusNorm(m *Matrix) float64 {
	sum := 0.0
	for _, v := range m.Data {
		sum += v * v
	}
	return math.Sqrt(sum)
}

// frobeniusError returns ‖a - b‖_F / ‖b‖_F (relative error).
func frobeniusError(a, b *Matrix) float64 {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		panic("shape mismatch in frobeniusError")
	}
	diff := NewMatrix(a.Rows, a.Cols)
	for i := range a.Data {
		diff.Data[i] = a.Data[i] - b.Data[i]
	}
	denom := frobeniusNorm(b)
	if denom == 0 {
		return frobeniusNorm(diff)
	}
	return frobeniusNorm(diff) / denom
}

// ─────────────────────────────────────────────
// SECTION 3: Matrix operations
// ─────────────────────────────────────────────

// MatMul computes C = A × B.
func MatMul(A, B *Matrix) *Matrix {
	if A.Cols != B.Rows {
		panic(fmt.Sprintf("MatMul shape mismatch: A%s × B%s", A.Shape(), B.Shape()))
	}
	C := NewMatrix(A.Rows, B.Cols)
	for i := 0; i < A.Rows; i++ {
		for k := 0; k < A.Cols; k++ {
			aik := A.At(i, k)
			for j := 0; j < B.Cols; j++ {
				C.Data[i*C.Cols+j] += aik * B.At(k, j)
			}
		}
	}
	return C
}

// MatAdd computes C = A + B (element-wise, in-place into A).
func MatAdd(A, B *Matrix) *Matrix {
	if A.Rows != B.Rows || A.Cols != B.Cols {
		panic("MatAdd shape mismatch")
	}
	C := A.Clone()
	for i := range C.Data {
		C.Data[i] += B.Data[i]
	}
	return C
}

// GeLU applies the Gaussian Error Linear Unit element-wise.
// Using the exact formula: 0.5 * x * (1 + erf(x / sqrt(2)))
func GeLU(m *Matrix) *Matrix {
	out := NewMatrix(m.Rows, m.Cols)
	sqrt2 := math.Sqrt2
	for i, v := range m.Data {
		out.Data[i] = 0.5 * v * (1 + math.Erf(v/sqrt2))
	}
	return out
}

// AllReduce simulates a sum all-reduce across `numGPUs` partitions.
// In a real system this would be a collective over the network;
// here we just sum the slice of matrices and broadcast the result.
// Returns the single summed matrix and records one communication event.
func AllReduce(parts []*Matrix) (*Matrix, CommunicationEvent) {
	if len(parts) == 0 {
		panic("AllReduce: empty partition list")
	}
	result := parts[0].Clone()
	for _, p := range parts[1:] {
		result = MatAdd(result, p)
	}
	// comm volume = 2*(N-1)/N * data_size  (ring all-reduce); approximate as 2*size here
	bytesPerElem := 8 // float64
	volume := 2 * len(parts[0].Data) * bytesPerElem
	ev := CommunicationEvent{
		Op:          "AllReduce",
		NumGPUs:     len(parts),
		BytesPerGPU: volume,
	}
	return result, ev
}

// ─────────────────────────────────────────────
// SECTION 4: Partitioning helpers
// ─────────────────────────────────────────────

// SplitColumns splits matrix M into `n` column-wise partitions.
// M has shape (r, c); each partition has shape (r, c/n).
func SplitColumns(M *Matrix, n int) []*Matrix {
	if M.Cols%n != 0 {
		panic(fmt.Sprintf("SplitColumns: cols %d not divisible by %d", M.Cols, n))
	}
	w := M.Cols / n
	parts := make([]*Matrix, n)
	for p := 0; p < n; p++ {
		parts[p] = NewMatrix(M.Rows, w)
		for r := 0; r < M.Rows; r++ {
			for c := 0; c < w; c++ {
				parts[p].Set(r, c, M.At(r, p*w+c))
			}
		}
	}
	return parts
}

// SplitRows splits matrix M into `n` row-wise partitions.
// M has shape (r, c); each partition has shape (r/n, c).
func SplitRows(M *Matrix, n int) []*Matrix {
	if M.Rows%n != 0 {
		panic(fmt.Sprintf("SplitRows: rows %d not divisible by %d", M.Rows, n))
	}
	h := M.Rows / n
	parts := make([]*Matrix, n)
	for p := 0; p < n; p++ {
		parts[p] = NewMatrix(h, M.Cols)
		for r := 0; r < h; r++ {
			for c := 0; c < M.Cols; c++ {
				parts[p].Set(r, c, M.At(p*h+r, c))
			}
		}
	}
	return parts
}

// ConcatColumns concatenates matrices horizontally: [M1 | M2 | ...] → (r, sum(c_i))
func ConcatColumns(parts []*Matrix) *Matrix {
	rows := parts[0].Rows
	totalCols := 0
	for _, p := range parts {
		totalCols += p.Cols
	}
	out := NewMatrix(rows, totalCols)
	col := 0
	for _, p := range parts {
		for r := 0; r < rows; r++ {
			for c := 0; c < p.Cols; c++ {
				out.Set(r, col+c, p.At(r, c))
			}
		}
		col += p.Cols
	}
	return out
}

// ConcatRows concatenates matrices vertically.
func ConcatRows(parts []*Matrix) *Matrix {
	cols := parts[0].Cols
	totalRows := 0
	for _, p := range parts {
		totalRows += p.Rows
	}
	out := NewMatrix(totalRows, cols)
	row := 0
	for _, p := range parts {
		for r := 0; r < p.Rows; r++ {
			for c := 0; c < cols; c++ {
				out.Set(row+r, c, p.At(r, c))
			}
		}
		row += p.Rows
	}
	return out
}

// ─────────────────────────────────────────────
// SECTION 5: GEMM class with metadata
// ─────────────────────────────────────────────

// CommunicationEvent records one collective operation.
type CommunicationEvent struct {
	Op          string // e.g. "AllReduce"
	NumGPUs     int
	BytesPerGPU int // bytes per GPU
}

func (e CommunicationEvent) String() string {
	return fmt.Sprintf("%s(gpus=%d, bytes/gpu=%d)", e.Op, e.NumGPUs, e.BytesPerGPU)
}

// GEMM represents one matrix-multiply layer in the MLP.
type GEMM struct {
	Name   string
	Split  SplitMethod
	NumGPU int

	// Inputs (set before Run)
	X *Matrix // input  (rows=batch, cols=d_in)
	A *Matrix // weight (rows=d_in, cols=d_out)

	// Outputs (set after Run)
	Output     *Matrix
	CommEvents []CommunicationEvent
	SyncPoints int // number of synchronization barriers required
}

func NewGEMM(name string, split SplitMethod, numGPU int) *GEMM {
	return &GEMM{Name: name, Split: split, NumGPU: numGPU}
}

// ─────────────────────────────────────────────
// SECTION 6: MLP Strategy implementations
// ─────────────────────────────────────────────

// MLPResult captures all metrics for one end-to-end MLP forward pass.
type MLPResult struct {
	StrategyName   string
	Description    string
	Output         *Matrix
	CommEvents     []CommunicationEvent
	SyncPoints     int
	TotalCommBytes int     // sum over all comm events × numGPUs
	RelativeError  float64 // vs ground truth
	Correct        bool    // error < threshold
}

func (r *MLPResult) TotalCommMB() float64 {
	return float64(r.TotalCommBytes) / (1024 * 1024)
}

// groundTruth computes the reference output on a single GPU: GeLU(X·A1) · B
func groundTruth(X, A, B *Matrix) *Matrix {
	Y := GeLU(MatMul(X, A))
	return MatMul(Y, B)
}

// ── Strategy 1 (Paper optimal): Column-split A, row-split B ──────────────────
//
//	GPU_i receives: X (full), A_i (columns), B_i (rows)
//	Forward:
//	  Y_i = GeLU(X · A_i)          ← no sync needed before GeLU
//	  Z_i = Y_i · B_i              ← each GPU holds partial output
//	  Z   = AllReduce(Z_0,...,Z_n) ← single all-reduce
//
// This is Figure 3a of the Megatron-LM paper.
func strategyPaperOptimal(X, A, B *Matrix, numGPU int) MLPResult {
	events := []CommunicationEvent{}
	syncPts := 0

	// Partition weights
	As := SplitColumns(A, numGPU) // A split by columns → (d_in, d_out/n)
	Bs := SplitRows(B, numGPU)    // B split by rows    → (d_out/n, d_out2)

	Zparts := make([]*Matrix, numGPU)
	for i := 0; i < numGPU; i++ {
		Yi := GeLU(MatMul(X, As[i])) // X is broadcast to every GPU
		Zparts[i] = MatMul(Yi, Bs[i])
	}

	// Single all-reduce in forward pass
	Z, ev := AllReduce(Zparts)
	events = append(events, ev)
	syncPts++ // g operator: one barrier

	totalBytes := 0
	for _, e := range events {
		totalBytes += e.BytesPerGPU * e.NumGPUs
	}
	return MLPResult{
		StrategyName:   "Paper Optimal (col-split A, row-split B)",
		Description:    "Column-split GEMM1 → independent GeLU → row-split GEMM2 → 1 AllReduce",
		Output:         Z,
		CommEvents:     events,
		SyncPoints:     syncPts,
		TotalCommBytes: totalBytes,
	}
}

// ── Strategy 2 (Row-split A, col-split X): requires sync BEFORE GeLU ─────────
//
//	GPU_i receives: X_i (columns of X), A_i (rows of A)
//	Forward:
//	  P_i = X_i · A_i              ← each partial product has shape (batch, d_out)
//	  P   = AllReduce(P_0,...,P_n) ← MUST sync before GeLU (non-linear!)
//	  Y   = GeLU(P)
//	  Z   = Y · B                  ← B not split; each GPU holds the whole B
//	                                  → needs a second all-reduce for the final Z,
//	                                     or B is replicated (common in practice)
//	  Z   = AllReduce(Z_0,...Z_n) ← or use replicated B and no second allreduce
//	                                 We model the worst case: 2 all-reduces.
//
// This corresponds to the "row-split" option the paper explicitly rejects.
func strategyRowSplitA(X, A, B *Matrix, numGPU int) MLPResult {
	events := []CommunicationEvent{}
	syncPts := 0

	Xs := SplitColumns(X, numGPU) // X split by columns → (batch, d_in/n)
	As := SplitRows(A, numGPU)    // A split by rows    → (d_in/n, d_out)

	// Each GPU computes its partial product
	Pparts := make([]*Matrix, numGPU)
	for i := 0; i < numGPU; i++ {
		Pparts[i] = MatMul(Xs[i], As[i])
	}

	// *** Mandatory sync BEFORE GeLU ***
	P, ev1 := AllReduce(Pparts)
	events = append(events, ev1)
	syncPts++

	Y := GeLU(P)

	// Now multiply by B; B is replicated on each GPU in this scheme.
	// Each GPU computes Y · B independently → result is the same on all GPUs.
	// No second AllReduce needed because B is replicated (but B must be fully available).
	// We record that B must be replicated (communication overhead at model init time).
	Z := MatMul(Y, B)

	totalBytes := 0
	for _, e := range events {
		totalBytes += e.BytesPerGPU * e.NumGPUs
	}
	return MLPResult{
		StrategyName:   "Row-split A (sync before GeLU)",
		Description:    "Row-split GEMM1 → AllReduce → GeLU → full B replicated → no 2nd AllReduce",
		Output:         Z,
		CommEvents:     events,
		SyncPoints:     syncPts,
		TotalCommBytes: totalBytes,
	}
}

// ── Strategy 3 (Naive: col-split both GEMMs) ──────────────────────────────────
//
// Split BOTH A and B by columns.
//
//	GEMM1 (col-split A):  GPU_i: Y_i = GeLU(X · A_i)  shape (batch, d_hidden/n)
//	GEMM2 (col-split B):  B_i has shape (d_hidden, d_out/n)
//
// For GPU_i to compute Y_i · B_i it needs the FULL Y (shape batch × d_hidden),
// not just its local Y_i (shape batch × d_hidden/n).  So we must AllGather Y
// before GEMM2, then each GPU independently computes its Z_i = Y · B_i.
// The final Z is assembled by ConcatColumns — no second collective needed.
// Total: 1 AllGather (same bytes as AllReduce) + 1 ConcatColumns (free locally).
func strategyColSplitBoth(X, A, B *Matrix, numGPU int) MLPResult {
	events := []CommunicationEvent{}
	syncPts := 0

	As := SplitColumns(A, numGPU) // (d_in, d_hidden/n)
	Bs := SplitColumns(B, numGPU) // (d_hidden, d_out/n)

	// GPU i: Y_i = GeLU(X · A_i) → shape (batch, d_hidden/n)
	Yparts := make([]*Matrix, numGPU)
	for i := 0; i < numGPU; i++ {
		Yparts[i] = GeLU(MatMul(X, As[i]))
	}

	// AllGather Y_parts → full Y (batch, d_hidden) — each GPU now has full Y
	Y := ConcatColumns(Yparts)
	evGather := CommunicationEvent{
		Op:          "AllGather",
		NumGPUs:     numGPU,
		BytesPerGPU: 2 * len(Yparts[0].Data) * 8,
	}
	events = append(events, evGather)
	syncPts++

	// GPU i: Z_i = Y · B_i → shape (batch, d_out/n); no further comm needed
	Zparts := make([]*Matrix, numGPU)
	for i := 0; i < numGPU; i++ {
		Zparts[i] = MatMul(Y, Bs[i])
	}

	// Local concat (no network) to assemble final output
	Z := ConcatColumns(Zparts)

	totalBytes := 0
	for _, e := range events {
		totalBytes += e.BytesPerGPU * e.NumGPUs
	}
	return MLPResult{
		StrategyName:   "Col-split both GEMMs",
		Description:    "Col-split GEMM1 → GeLU → col-split GEMM2 → AllGather",
		Output:         Z,
		CommEvents:     events,
		SyncPoints:     syncPts,
		TotalCommBytes: totalBytes,
	}
}

// ── Strategy 4 (Naive: row-split both GEMMs) ─────────────────────────────────
//
// Split A by rows and B by rows.  GEMM1 row-split forces sync before GeLU,
// then GEMM2 row-split requires another all-reduce → 2 all-reduces total.
func strategyRowSplitBoth(X, A, B *Matrix, numGPU int) MLPResult {
	events := []CommunicationEvent{}
	syncPts := 0

	Xs := SplitColumns(X, numGPU)
	As := SplitRows(A, numGPU)

	Pparts := make([]*Matrix, numGPU)
	for i := 0; i < numGPU; i++ {
		Pparts[i] = MatMul(Xs[i], As[i])
	}
	P, ev1 := AllReduce(Pparts)
	events = append(events, ev1)
	syncPts++

	Y := GeLU(P)

	// Now row-split B and Y accordingly
	Bs := SplitRows(B, numGPU)
	Ys := SplitColumns(Y, numGPU) // split Y col-wise to match row-split of B

	Zparts := make([]*Matrix, numGPU)
	for i := 0; i < numGPU; i++ {
		Zparts[i] = MatMul(Ys[i], Bs[i])
	}
	Z, ev2 := AllReduce(Zparts)
	events = append(events, ev2)
	syncPts++

	totalBytes := 0
	for _, e := range events {
		totalBytes += e.BytesPerGPU * e.NumGPUs
	}
	return MLPResult{
		StrategyName:   "Row-split both GEMMs",
		Description:    "Row-split GEMM1 → AllReduce → GeLU → Row-split GEMM2 → AllReduce",
		Output:         Z,
		CommEvents:     events,
		SyncPoints:     syncPts,
		TotalCommBytes: totalBytes,
	}
}

// ── Strategy 5 (Swapped: row-split A, col-split B) ────────────────────────────
//
// Row-split A (requires all-reduce before GeLU), then col-split B.
// After AllReduce we have full Y; col-split B means we need AllGather at end.
// → 2 sync points, same as strategy 4.
func strategySwapped(X, A, B *Matrix, numGPU int) MLPResult {
	events := []CommunicationEvent{}
	syncPts := 0

	Xs := SplitColumns(X, numGPU)
	As := SplitRows(A, numGPU)

	Pparts := make([]*Matrix, numGPU)
	for i := 0; i < numGPU; i++ {
		Pparts[i] = MatMul(Xs[i], As[i])
	}
	P, ev1 := AllReduce(Pparts)
	events = append(events, ev1)
	syncPts++

	Y := GeLU(P)

	Bs := SplitColumns(B, numGPU)
	Zparts := make([]*Matrix, numGPU)
	for i := 0; i < numGPU; i++ {
		Zparts[i] = MatMul(Y, Bs[i])
	}
	Z := ConcatColumns(Zparts)
	ev2 := CommunicationEvent{
		Op:          "AllGather",
		NumGPUs:     numGPU,
		BytesPerGPU: 2 * len(Zparts[0].Data) * 8,
	}
	events = append(events, ev2)
	syncPts++

	totalBytes := 0
	for _, e := range events {
		totalBytes += e.BytesPerGPU * e.NumGPUs
	}
	return MLPResult{
		StrategyName:   "Swapped (row-split A, col-split B)",
		Description:    "Row-split GEMM1 → AllReduce → GeLU → col-split GEMM2 → AllGather",
		Output:         Z,
		CommEvents:     events,
		SyncPoints:     syncPts,
		TotalCommBytes: totalBytes,
	}
}

// ── Strategy 6: No split (single GPU baseline) ────────────────────────────────
func strategyNoSplit(X, A, B *Matrix) MLPResult {
	Y := GeLU(MatMul(X, A))
	Z := MatMul(Y, B)
	return MLPResult{
		StrategyName:   "No Split (single GPU baseline)",
		Description:    "Full computation on one GPU, no communication",
		Output:         Z,
		CommEvents:     nil,
		SyncPoints:     0,
		TotalCommBytes: 0,
	}
}

// ─────────────────────────────────────────────
// SECTION 7: Experiment runner & reporting
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
		strategyNoSplit(X, A, B),
		strategyPaperOptimal(X, A, B, cfg.NumGPU),
		strategyRowSplitA(X, A, B, cfg.NumGPU),
		strategyColSplitBoth(X, A, B, cfg.NumGPU),
		strategyRowSplitBoth(X, A, B, cfg.NumGPU),
		strategySwapped(X, A, B, cfg.NumGPU),
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
		fmt.Printf("    Sync points      : %d\n", r.SyncPoints)
		fmt.Printf("    Total comm data  : %.3f MB\n", r.TotalCommMB())
		fmt.Printf("    Relative error   : %.2e  ", r.RelativeError)
		if r.Correct {
			fmt.Println("✓ CORRECT")
		} else {
			fmt.Println("✗ NUMERICALLY WRONG (non-linear barrier missing!)")
		}
	}

	// Summary comparison table
	fmt.Println()
	fmt.Println(strings.Repeat("═", 100))
	fmt.Println("  SUMMARY TABLE")
	fmt.Println(strings.Repeat("═", 100))
	fmt.Printf("  %-42s  %-6s  %-12s  %-12s  %s\n",
		"Strategy", "Syncs", "Comm (MB)", "Rel. Error", "Status")
	fmt.Println("  " + strings.Repeat("-", 96))
	for _, r := range results {
		status := "✓ OK"
		if !r.Correct {
			status = "✗ WRONG"
		}
		fmt.Printf("  %-42s  %-6d  %-12.3f  %-12.2e  %s\n",
			r.StrategyName, r.SyncPoints, r.TotalCommMB(), r.RelativeError, status)
	}
	fmt.Println(strings.Repeat("═", 100))

	// Analysis
	fmt.Println()
	fmt.Println("  ANALYSIS")
	fmt.Println(sep)
	// 	fmt.Println(`
	//   The Megatron-LM paper selects "col-split A, row-split B" (Strategy 2 above) for the
	//   following reasons, all confirmed by the metrics above:

	//   1. CORRECTNESS WITHOUT EXTRA SYNC
	//      Column-splitting A means each GPU computes GeLU(X·A_i) independently. Because
	//      GeLU is applied to a complete partial result (not a partial sum), no synchronization
	//      is required before the activation. Row-splitting A forces each GPU to contribute
	//      a partial sum X_i·A_i that must be all-reduced before GeLU can be applied (since
	//      GeLU(sum) ≠ sum(GeLU)). Strategies that skip this barrier produce wrong outputs.

	//   2. MINIMUM SYNCHRONIZATION: ONE ALL-REDUCE PER FORWARD PASS
	//      The paper-optimal strategy (col-split A, row-split B) requires exactly 1 AllReduce
	//      in the forward pass and 1 in the backward pass. Every other multi-GPU strategy
	//      in this experiment requires 2 sync points or incurs a replication cost.

	//   3. COMMUNICATION VOLUME
	//      With 1 AllReduce the paper strategy moves the least data across the interconnect.
	//      Strategies with 2 collective operations move roughly twice the data.

	//  4. WEIGHT MEMORY IS PERFECTLY SHARDED
	//     Column-splitting A and row-splitting B distributes weights evenly across GPUs with
	//     no replication. Row-split-A strategies often require B to be fully replicated on
	//     every GPU to avoid a second all-reduce, wasting memory.
	//
	// `)
}

// ─────────────────────────────────────────────
// SECTION 8: main — tunable experiments
// ─────────────────────────────────────────────

func main() {
	// ── Experiment A: small, easy to follow ──────────────────────────────────
	runExperiment(ExperimentConfig{
		BatchSize: 4,
		DIn:       8,
		DHidden:   16,
		DOut:      8,
		NumGPU:    2,
		Seed:      42,
	})

	// ── Experiment B: larger hidden dim, 4 GPUs ───────────────────────────────
	runExperiment(ExperimentConfig{
		BatchSize: 8,
		DIn:       64,
		DHidden:   256,
		DOut:      64,
		NumGPU:    4,
		Seed:      7,
	})

	// ── Experiment C: near transformer scale, 8 GPUs ─────────────────────────
	runExperiment(ExperimentConfig{
		BatchSize: 16,
		DIn:       512,
		DHidden:   2048,
		DOut:      512,
		NumGPU:    8,
		Seed:      99,
	})
}
