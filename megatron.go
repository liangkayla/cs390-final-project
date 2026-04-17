package cs390finalproject

import (
	"fmt"
	"math/rand"
	"sync"
	"time"
)

// Matrix is a flat row-major 2D matrix
type Matrix struct {
	rows, cols int
	data       []float64
}

func NewMatrix(rows, cols int) *Matrix {
	return &Matrix{rows: rows, cols: cols, data: make([]float64, rows*cols)}
}

func (m *Matrix) At(r, c int) float64      { return m.data[r*m.cols+c] }
func (m *Matrix) Set(r, c int, v float64)  { m.data[r*m.cols+c] = v }

func RandomMatrix(rows, cols int) *Matrix {
	m := NewMatrix(rows, cols)
	for i := range m.data {
		m.data[i] = rand.Float64()
	}
	return m
}

// multiplyShardRows multiplies rows [startRow, endRow) of A x B into C
func multiplyShardRows(A, B, C *Matrix, startRow, endRow, shardID int, wg *sync.WaitGroup) {
	defer wg.Done()
	fmt.Printf("  [Shard %d] computing rows %d..%d\n", shardID, startRow, endRow-1)

	for i := startRow; i < endRow; i++ {
		for j := 0; j < B.cols; j++ {
			sum := 0.0
			for k := 0; k < A.cols; k++ {
				sum += A.At(i, k) * B.At(k, j)
			}
			C.Set(i, j, sum)
		}
	}
}

// ParallelMatMul splits A x B across `numShards` goroutines (simulated GPUs)
func ParallelMatMul(A, B *Matrix, numShards int) *Matrix {
	if A.cols != B.rows {
		panic("incompatible matrix dimensions")
	}
	C := NewMatrix(A.rows, B.cols)

	rowsPerShard := (A.rows + numShards - 1) / numShards // ceiling division

	var wg sync.WaitGroup
	for s := 0; s < numShards; s++ {
		startRow := s * rowsPerShard
		endRow := startRow + rowsPerShard
		if endRow > A.rows {
			endRow = A.rows
		}
		if startRow >= A.rows {
			break // no work left
		}
		wg.Add(1)
		go multiplyShardRows(A, B, C, startRow, endRow, s, &wg)
	}
	wg.Wait()
	return C
}

func main() {
	rand.Seed(time.Now().UnixNano())

	N := 512
	numShards := 4

	fmt.Printf("Multiplying %dx%d matrices across %d simulated GPU shards\n\n", N, N, numShards)

	A := RandomMatrix(N, N)
	B := RandomMatrix(N, N)

	start := time.Now()
	_ = ParallelMatMul(A, B, numShards)
	elapsed := time.Since(start)

	fmt.Printf("\nDone in %v\n", elapsed)
}