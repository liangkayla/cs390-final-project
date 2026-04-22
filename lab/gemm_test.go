package main

import (
	"math/rand"
	"testing"

	"gonum.org/v1/gonum/mat"
)

// ─────────────────────────────────────────────
// Test helpers
// ─────────────────────────────────────────────

const testErrThreshold = 1e-9

// expectCorrect verifies that result.Output numerically matches the reference
// output (ground truth computed on a single GPU).
func expectCorrect(t *testing.T, name string, result MLPResult, X, A, B interface{}) {
	t.Helper()
	if !result.Correct {
		t.Fatalf("%s: output is numerically incorrect (relative error %.2e exceeds threshold %.2e). "+
			"Did you apply GeLU at the right point?", name, result.RelativeError, testErrThreshold)
	}
}

// expectSyncPoints checks that a strategy uses exactly the expected number of
// synchronisation points (AllReduce calls).
func expectSyncPoints(t *testing.T, name string, result MLPResult, expected int) {
	t.Helper()
	if result.SyncPoints != expected {
		t.Fatalf("%s: expected %d sync point(s), got %d", name, expected, result.SyncPoints)
	}
}

// expectCommEvents checks that a strategy records exactly the expected number
// of communication events.
func expectCommEvents(t *testing.T, name string, result MLPResult, expected int) {
	t.Helper()
	if len(result.CommEvents) != expected {
		t.Fatalf("%s: expected %d communication event(s), got %d", name, expected, len(result.CommEvents))
	}
}

// expectOutputShape checks that the output matrix has the right shape.
func expectOutputShape(t *testing.T, name string, result MLPResult, rows, cols int) {
	t.Helper()
	r, c := result.Output.Dims()
	if r != rows || c != cols {
		t.Fatalf("%s: expected output shape (%d×%d), got (%d×%d)", name, rows, cols, r, c)
	}
}

// expectFLOPs checks that the reported FLOPs are within 1% of the expected value.
func expectFLOPs(t *testing.T, name string, result MLPResult, expectedFLOPs float64) {
	t.Helper()
	ratio := result.Metrics.FLOPs / expectedFLOPs
	if ratio < 0.99 || ratio > 1.01 {
		t.Fatalf("%s: expected ~%.3e FLOPs, got %.3e (ratio %.3f)", name, expectedFLOPs, result.Metrics.FLOPs, ratio)
	}
}

// expectCommOpName checks that all AllReduce events carry the right operation name.
func expectCommOpName(t *testing.T, name string, result MLPResult, op string) {
	t.Helper()
	for i, e := range result.CommEvents {
		if e.Op != op {
			t.Fatalf("%s: comm event %d has Op=%q, expected %q", name, i, e.Op, op)
		}
	}
}

// makeMatrices produces deterministic random matrices for tests.
func makeMatrices(batchSize, dIn, dHidden, dOut int, seed int64) (X, A, B interface{}) {
	return nil, nil, nil // placeholder; tests use NewRandMatrix directly
}

// ─────────────────────────────────────────────
// TestNoSplit
//
// The baseline single-GPU strategy must produce the ground-truth output,
// report zero communication, and account for all FLOPs.
// ─────────────────────────────────────────────

func TestNoSplit(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	X := NewRandMatrix(4, 8, rng)
	A := NewRandMatrix(8, 16, rng)
	B := NewRandMatrix(16, 8, rng)

	ref := groundTruth(X, A, B)
	result := strategyNoSplit(X, A, B, 1)

	// Mark correctness manually (strategyNoSplit does not set it).
	result.RelativeError = frobeniusError(result.Output, ref)
	result.Correct = result.RelativeError < testErrThreshold

	expectCorrect(t, "NoSplit correctness", result, nil, nil, nil)
	expectOutputShape(t, "NoSplit output shape", result, 4, 8)
	expectCommEvents(t, "NoSplit no communication", result, 0)
	expectSyncPoints(t, "NoSplit no sync points", result, 0)

	// FLOPs: GEMM1 (4×16×8=1024) + GEMM2 (4×8×16=1024) → 2048 total multiply-adds → 4096 FLOPs
	expectedFLOPs := gemmFLOPs(4, 16, 8) + gemmFLOPs(4, 8, 16)
	expectFLOPs(t, "NoSplit FLOPs", result, expectedFLOPs)
}

// ─────────────────────────────────────────────
// TestPaperOptimal_SmallTwoGPU
//
// 2 GPUs, tiny matrices.  The optimal (column-split) strategy must produce
// the correct answer and use exactly ONE AllReduce at the end.
// ─────────────────────────────────────────────

func TestPaperOptimal_SmallTwoGPU(t *testing.T) {
	const numGPU = 2
	rng := rand.New(rand.NewSource(7))
	X := NewRandMatrix(4, 8, rng)
	A := NewRandMatrix(8, 16, rng) // 16 cols → 8 per GPU
	B := NewRandMatrix(16, 8, rng) // 16 rows → 8 per GPU

	ref := groundTruth(X, A, B)
	result := strategyPaperOptimal(X, A, B, numGPU)
	result.RelativeError = frobeniusError(result.Output, ref)
	result.Correct = result.RelativeError < testErrThreshold

	expectCorrect(t, "PaperOptimal (2 GPU) correctness", result, nil, nil, nil)
	expectOutputShape(t, "PaperOptimal (2 GPU) output shape", result, 4, 8)

	// Must have exactly one communication event (the single AllReduce at the end).
	expectCommEvents(t, "PaperOptimal (2 GPU) comm events", result, 1)
	expectSyncPoints(t, "PaperOptimal (2 GPU) sync points", result, 1)
	expectCommOpName(t, "PaperOptimal (2 GPU) comm op", result, "AllReduce")
}

// ─────────────────────────────────────────────
// TestPaperOptimal_FourGPU
//
// 4 GPUs, medium-sized matrices.  Verifies correctness and communication
// structure scale correctly with more GPUs.
// ─────────────────────────────────────────────

func TestPaperOptimal_FourGPU(t *testing.T) {
	const numGPU = 4
	rng := rand.New(rand.NewSource(99))
	X := NewRandMatrix(8, 64, rng)
	A := NewRandMatrix(64, 256, rng) // 256 cols → 64 per GPU
	B := NewRandMatrix(256, 64, rng) // 256 rows → 64 per GPU

	ref := groundTruth(X, A, B)
	result := strategyPaperOptimal(X, A, B, numGPU)
	result.RelativeError = frobeniusError(result.Output, ref)
	result.Correct = result.RelativeError < testErrThreshold

	expectCorrect(t, "PaperOptimal (4 GPU) correctness", result, nil, nil, nil)
	expectOutputShape(t, "PaperOptimal (4 GPU) output shape", result, 8, 64)
	expectCommEvents(t, "PaperOptimal (4 GPU) comm events", result, 1)
	expectSyncPoints(t, "PaperOptimal (4 GPU) sync points", result, 1)
}

// ─────────────────────────────────────────────
// TestPaperOptimal_FLOPs
//
// FLOPs reported by the optimal strategy must equal the total work across
// all GPUs, which equals the single-GPU FLOPs (parallelism doesn't change
// total arithmetic).
// ─────────────────────────────────────────────

func TestPaperOptimal_FLOPs(t *testing.T) {
	const numGPU = 2
	rng := rand.New(rand.NewSource(13))
	m, k, h, c := 4, 8, 16, 8
	X := NewRandMatrix(m, k, rng)
	A := NewRandMatrix(k, h, rng)
	B := NewRandMatrix(h, c, rng)

	result := strategyPaperOptimal(X, A, B, numGPU)

	// Each GPU does (m × h/n × k) + (m × c × h/n) multiply-adds → ×2 for FLOPs.
	// Summed over n GPUs this equals the single-GPU total.
	expectedFLOPs := gemmFLOPs(m, h/numGPU, k)*float64(numGPU) +
		gemmFLOPs(m, c, h/numGPU)*float64(numGPU)
	expectFLOPs(t, "PaperOptimal FLOPs", result, expectedFLOPs)
}

// ─────────────────────────────────────────────
// TestPaperSuboptimal_SmallTwoGPU
//
// 2 GPUs, tiny matrices.  The suboptimal (row-split) strategy must produce
// the correct answer and use TWO sync points (one mid-block AllReduce plus
// the backward-pass conjugate counted in the initial syncPts=1).
// ─────────────────────────────────────────────

func TestPaperSuboptimal_SmallTwoGPU(t *testing.T) {
	const numGPU = 2
	rng := rand.New(rand.NewSource(7))
	X := NewRandMatrix(4, 8, rng)  // 8 cols → 4 per GPU
	A := NewRandMatrix(8, 16, rng) // 8 rows → 4 per GPU
	B := NewRandMatrix(16, 8, rng) // 8 cols → 4 per GPU

	ref := groundTruth(X, A, B)
	result := strategyPaperSuboptimal(X, A, B, numGPU)
	result.RelativeError = frobeniusError(result.Output, ref)
	result.Correct = result.RelativeError < testErrThreshold

	expectCorrect(t, "PaperSuboptimal (2 GPU) correctness", result, nil, nil, nil)
	expectOutputShape(t, "PaperSuboptimal (2 GPU) output shape", result, 4, 8)

	// Mid-block AllReduce before GeLU is the key communication event.
	expectCommEvents(t, "PaperSuboptimal (2 GPU) comm events", result, 1)
	// syncPts=2: initial 1 (backward-pass f) + 1 increment for the mid-block AllReduce.
	expectSyncPoints(t, "PaperSuboptimal (2 GPU) sync points", result, 2)
	expectCommOpName(t, "PaperSuboptimal (2 GPU) comm op", result, "AllReduce")
}

// ─────────────────────────────────────────────
// TestPaperSuboptimal_FourGPU
//
// 4 GPUs, medium-sized matrices.
// ─────────────────────────────────────────────

func TestPaperSuboptimal_FourGPU(t *testing.T) {
	const numGPU = 4
	rng := rand.New(rand.NewSource(99))
	X := NewRandMatrix(8, 64, rng)
	A := NewRandMatrix(64, 256, rng)
	B := NewRandMatrix(256, 64, rng)

	ref := groundTruth(X, A, B)
	result := strategyPaperSuboptimal(X, A, B, numGPU)
	result.RelativeError = frobeniusError(result.Output, ref)
	result.Correct = result.RelativeError < testErrThreshold

	expectCorrect(t, "PaperSuboptimal (4 GPU) correctness", result, nil, nil, nil)
	expectOutputShape(t, "PaperSuboptimal (4 GPU) output shape", result, 8, 64)
	expectCommEvents(t, "PaperSuboptimal (4 GPU) comm events", result, 1)
	expectSyncPoints(t, "PaperSuboptimal (4 GPU) sync points", result, 2)
}

// ─────────────────────────────────────────────
// TestPaperSuboptimal_FLOPs
//
// The reported FLOPs for the suboptimal strategy must equal the total
// arithmetic across all GPUs (same as optimal — communication overhead
// differs, not raw compute).
// ─────────────────────────────────────────────

func TestPaperSuboptimal_FLOPs(t *testing.T) {
	const numGPU = 2
	rng := rand.New(rand.NewSource(13))
	m, k, h, c := 4, 8, 16, 8
	X := NewRandMatrix(m, k, rng)
	A := NewRandMatrix(k, h, rng)
	B := NewRandMatrix(h, c, rng)

	result := strategyPaperSuboptimal(X, A, B, numGPU)

	// GEMM1: each GPU does (m × h × k/n) multiply-adds → gemmFLOPs(m, h, k/n)
	// GEMM2: each GPU does (m × c/n × h)  multiply-adds → gemmFLOPs(m, c/n, h)
	expectedFLOPs := gemmFLOPs(m, h, k/numGPU)*float64(numGPU) +
		gemmFLOPs(m, c/numGPU, h)*float64(numGPU)
	expectFLOPs(t, "PaperSuboptimal FLOPs", result, expectedFLOPs)
}

// ─────────────────────────────────────────────
// TestBothStrategiesAgreeWithBaseline
//
// End-to-end sanity check: on all experiment configurations, both parallel
// strategies must produce an output that agrees with the single-GPU baseline
// to within the numerical threshold.
// ─────────────────────────────────────────────

func TestBothStrategiesAgreeWithBaseline(t *testing.T) {
	configs := []struct {
		name          string
		m, k, h, c, n int
		seed          int64
	}{
		{"tiny-2gpu", 4, 8, 16, 8, 2, 1},
		{"small-4gpu", 8, 64, 256, 64, 4, 7},
		{"medium-8gpu", 8, 512, 2048, 512, 8, 99},
	}

	for _, cfg := range configs {
		cfg := cfg
		t.Run(cfg.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(cfg.seed))
			X := NewRandMatrix(cfg.m, cfg.k, rng)
			A := NewRandMatrix(cfg.k, cfg.h, rng)
			B := NewRandMatrix(cfg.h, cfg.c, rng)

			ref := groundTruth(X, A, B)

			opt := strategyPaperOptimal(X, A, B, cfg.n)
			opt.RelativeError = frobeniusError(opt.Output, ref)
			opt.Correct = opt.RelativeError < testErrThreshold

			sub := strategyPaperSuboptimal(X, A, B, cfg.n)
			sub.RelativeError = frobeniusError(sub.Output, ref)
			sub.Correct = sub.RelativeError < testErrThreshold

			expectCorrect(t, cfg.name+" optimal", opt, nil, nil, nil)
			expectCorrect(t, cfg.name+" suboptimal", sub, nil, nil, nil)
		})
	}
}

// ─────────────────────────────────────────────
// TestOptimalFewerSyncsThanSuboptimal
//
// The optimal strategy must use strictly fewer sync points than the
// suboptimal strategy across all GPU counts.  This is the quantitative
// statement of why one strategy is preferred over the other.
// ─────────────────────────────────────────────

func TestOptimalFewerSyncsThanSuboptimal(t *testing.T) {
	for _, numGPU := range []int{2, 4, 8} {
		numGPU := numGPU
		t.Run("gpus", func(t *testing.T) {
			rng := rand.New(rand.NewSource(int64(numGPU)))
			X := NewRandMatrix(8, 64, rng)
			A := NewRandMatrix(64, 256, rng)
			B := NewRandMatrix(256, 64, rng)

			opt := strategyPaperOptimal(X, A, B, numGPU)
			sub := strategyPaperSuboptimal(X, A, B, numGPU)

			if opt.SyncPoints >= sub.SyncPoints {
				t.Fatalf("with %d GPUs: optimal syncPts=%d should be < suboptimal syncPts=%d",
					numGPU, opt.SyncPoints, sub.SyncPoints)
			}
		})
	}
}

// ─────────────────────────────────────────────
// TestSplitColumnsAndRows
//
// Unit tests for the partitioning helpers.  These are provided to help you
// debug your strategy implementations — no changes needed.
// ─────────────────────────────────────────────

func TestSplitColumnsAndRows(t *testing.T) {
	rng := rand.New(rand.NewSource(0))
	M := NewRandMatrix(4, 8, rng) // 4 rows, 8 cols

	// SplitColumns: 2 parts of shape (4, 4)
	colParts := SplitColumns(M, 2)
	if len(colParts) != 2 {
		t.Fatalf("SplitColumns returned %d parts, expected 2", len(colParts))
	}
	for i, p := range colParts {
		r, c := p.Dims()
		if r != 4 || c != 4 {
			t.Fatalf("SplitColumns part %d has shape (%d×%d), expected (4×4)", i, r, c)
		}
	}

	// ConcatColumns should reconstruct M exactly.
	reconstructed := ConcatColumns(colParts)
	if err := frobeniusError(reconstructed, M); err > 1e-14 {
		t.Fatalf("ConcatColumns(SplitColumns(M)) != M (error=%.2e)", err)
	}

	// SplitRows: 2 parts of shape (2, 8)
	rowParts := SplitRows(M, 2)
	if len(rowParts) != 2 {
		t.Fatalf("SplitRows returned %d parts, expected 2", len(rowParts))
	}
	for i, p := range rowParts {
		r, c := p.Dims()
		if r != 2 || c != 8 {
			t.Fatalf("SplitRows part %d has shape (%d×%d), expected (2×8)", i, r, c)
		}
	}

	// ConcatRows should reconstruct M exactly.
	reconstructed = ConcatRows(rowParts)
	if err := frobeniusError(reconstructed, M); err > 1e-14 {
		t.Fatalf("ConcatRows(SplitRows(M)) != M (error=%.2e)", err)
	}
}

// ─────────────────────────────────────────────
// TestAllReduce
//
// AllReduce on two identity-like matrices should return their element-wise sum.
// ─────────────────────────────────────────────

func TestAllReduce(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	A := NewRandMatrix(4, 4, rng)
	B := NewRandMatrix(4, 4, rng)

	result, ev := AllReduce([]*mat.Dense{A, B})

	expected := MatAdd(A, B)
	if err := frobeniusError(result, expected); err > 1e-14 {
		t.Fatalf("AllReduce sum incorrect (error=%.2e)", err)
	}
	if ev.Op != "AllReduce" {
		t.Fatalf("AllReduce event Op=%q, expected \"AllReduce\"", ev.Op)
	}
	if ev.NumGPUs != 2 {
		t.Fatalf("AllReduce event NumGPUs=%d, expected 2", ev.NumGPUs)
	}
}

// ─────────────────────────────────────────────
// TestGeLUNonLinearity
//
// Verifies that GeLU(a + b) ≠ GeLU(a) + GeLU(b) for non-trivial inputs.
// This test illustrates WHY the suboptimal strategy needs a mid-block sync:
// you cannot defer GeLU until after partial sums are computed independently.
// ─────────────────────────────────────────────

func TestGeLUNonLinearity(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	A := NewRandMatrix(4, 4, rng)
	B := NewRandMatrix(4, 4, rng)

	// GeLU(A + B) vs GeLU(A) + GeLU(B)
	sum := MatAdd(A, B)
	geluSum := GeLU(sum)                // correct
	sumGelu := MatAdd(GeLU(A), GeLU(B)) // wrong (what row-split would do without sync)

	err := frobeniusError(geluSum, sumGelu)
	if err < 1e-6 {
		t.Fatalf("GeLU appears linear (error=%.2e); test matrices may be degenerate", err)
	}
	// Passes if error is large — confirming non-linearity.
}
