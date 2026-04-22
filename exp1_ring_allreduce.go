// package cs390finalproject

// Experiment 1: Ring All-Reduce Simulation
//
// Models the communication pattern used in tensor parallelism (and data
// parallelism) in Megatron-LM. Each "GPU" is a goroutine holding a
// partition of a weight gradient vector. The ring all-reduce sums all
// partitions so every worker ends up with the globally-summed result.
//
// Algorithm (bandwidth-optimal ring all-reduce):
//   Phase 1 - Reduce-Scatter: N-1 rounds where each worker sends a chunk
//             to its right neighbor and accumulates from its left neighbor.
//             After this, each worker holds ONE fully-reduced chunk.
//   Phase 2 - All-Gather: N-1 rounds where each worker sends its reduced
//             chunk rightward. After this, every worker has ALL chunks.
//
// This is exactly what NCCL does between GPUs. We simulate the "latency"
// of each send with a configurable delay to model real NVLink/InfiniBand.
//
// Knobs to turn:
//   --workers    number of simulated GPUs (goroutines)
//   --msgsize    total gradient vector size (number of float64 elements)
//   --latency    simulated per-message latency in microseconds
//   --runs       number of timed runs to average
//
// Output (JSON to stdout): timing results suitable for graphing.

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"time"
)

// Result holds the measurement for one (workers, msgsize) configuration.
type Result struct {
	Workers       int     `json:"workers"`
	MsgSize       int     `json:"msg_size"`
	LatencyUS     int     `json:"latency_us"`
	WallTimeMS    float64 `json:"wall_time_ms"`
	ThroughputMBs float64 `json:"throughput_mbs"` // MB of useful gradient data / second
	IdealSpeedup  float64 `json:"ideal_speedup"`  // workers / actual_speedup (efficiency)
	ScalingPct    float64 `json:"scaling_pct"`
}

func main() {
	workers := flag.Int("workers", 0, "number of workers (0 = sweep 2,4,8,16)")
	msgSize := flag.Int("msgsize", 0, "gradient vector size in float64 elements (0 = sweep)")
	latencyUS := flag.Int("latency", 10, "simulated per-hop latency in microseconds")
	runs := flag.Int("runs", 5, "timed runs to average per config")
	flag.Parse()

	// Sweep configs if not pinned
	workerCounts := []int{2, 4, 8, 16}
	if *workers != 0 {
		workerCounts = []int{*workers}
	}
	msgSizes := []int{1_000, 10_000, 100_000, 1_000_000}
	if *msgSize != 0 {
		msgSizes = []int{*msgSize}
	}

	var results []Result

	// Establish baseline: time for 2 workers (minimum), used to compute relative scaling.
	// We compare how wall time grows as we add more workers relative to 2-worker baseline.
	baselineTimes := make(map[int]float64) // msgSize -> 2-worker time
	for _, m := range msgSizes {
		baselineTimes[m] = benchmarkRingAllReduce(2, m, *latencyUS, *runs)
	}

	for _, w := range workerCounts {
		for _, m := range msgSizes {
			wallMS := benchmarkRingAllReduce(w, m, *latencyUS, *runs)

			// Throughput: we moved msgSize float64s (8 bytes each) usefully
			dataBytes := float64(m * 8)
			throughput := dataBytes / (wallMS / 1000.0) / (1024 * 1024) // MB/s

			// Scaling efficiency vs 2-worker baseline:
			// Ideal: if we double workers, time halves (perfect linear scaling from baseline)
			// actual scaling = (baseline_time * 2 / w) / wallMS
			idealMS := baselineTimes[m] * (2.0 / float64(w))
			scaling := (idealMS / wallMS) * 100.0
			if w == 2 {
				scaling = 100.0
			}

			results = append(results, Result{
				Workers:       w,
				MsgSize:       m,
				LatencyUS:     *latencyUS,
				WallTimeMS:    wallMS,
				ThroughputMBs: throughput,
				IdealSpeedup:  float64(w) / 2.0,
				ScalingPct:    scaling,
			})

			fmt.Fprintf(os.Stderr, "workers=%d msg=%d -> %.2f ms (%.1f%% scaling efficiency)\n",
				w, m, wallMS, scaling)
		}
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(results)
}

// benchmarkSingleWorker measures time to "process" data without any comms.
func benchmarkSingleWorker(msgSize, runs int) float64 {
	data := makeGradient(msgSize)
	var total time.Duration
	for i := 0; i < runs; i++ {
		start := time.Now()
		_ = sumSlice(data) // simulate local reduction work
		total += time.Since(start)
	}
	return float64(total.Milliseconds()) / float64(runs)
}

// benchmarkRingAllReduce runs the ring all-reduce N times and returns avg ms.
func benchmarkRingAllReduce(numWorkers, msgSize, latencyUS, runs int) float64 {
	var total time.Duration
	for i := 0; i < runs; i++ {
		start := time.Now()
		runRingAllReduce(numWorkers, msgSize, latencyUS)
		total += time.Since(start)
	}
	return float64(total.Milliseconds()) / float64(runs)
}

// runRingAllReduce executes one full ring all-reduce across numWorkers goroutines.
// Each goroutine starts with a random gradient chunk of size msgSize/numWorkers.
func runRingAllReduce(numWorkers, msgSize, latencyUS int) {
	chunkSize := msgSize / numWorkers
	if chunkSize < 1 {
		chunkSize = 1
	}

	// Each worker owns numWorkers chunks; initially only chunk[workerID] is populated.
	// We use a [numWorkers][numWorkers][]float64 grid.
	grads := make([][][]float64, numWorkers)
	for i := range grads {
		grads[i] = make([][]float64, numWorkers)
		for j := range grads[i] {
			if j == i {
				grads[i][j] = makeGradient(chunkSize) // worker i "owns" chunk i
			} else {
				grads[i][j] = make([]float64, chunkSize)
			}
		}
	}

	// Channel ring: worker i sends to worker (i+1) % N
	// We create a channel between each consecutive pair.
	type msg struct {
		chunkID int
		data    []float64
	}
	channels := make([]chan msg, numWorkers)
	for i := range channels {
		channels[i] = make(chan msg, 1)
	}

	latency := time.Duration(latencyUS) * time.Microsecond
	var wg sync.WaitGroup

	// ---- Phase 1: Reduce-Scatter ----
	// Each worker sends chunk (workerID - step) mod N to right neighbor,
	// accumulates received chunk from left neighbor.
	wg.Add(numWorkers)
	for workerID := 0; workerID < numWorkers; workerID++ {
		go func(id int) {
			defer wg.Done()
			right := (id + 1) % numWorkers
			left := channels[id] // I receive from this channel

			sendChunk := id // start by sending my own chunk

			for step := 0; step < numWorkers-1; step++ {
				// Simulate network latency
				time.Sleep(latency)

				// Send current chunk to right neighbor
				dataCopy := make([]float64, len(grads[id][sendChunk]))
				copy(dataCopy, grads[id][sendChunk])
				channels[right] <- msg{chunkID: sendChunk, data: dataCopy}

				// Receive chunk from left neighbor and accumulate
				received := <-left
				for k, v := range received.data {
					grads[id][received.chunkID][k] += v
				}

				// Next step we forward the chunk we just received (minus 1)
				sendChunk = (sendChunk - 1 + numWorkers) % numWorkers
			}
		}(workerID)
	}
	wg.Wait()

	// ---- Phase 2: All-Gather ----
	// Each worker now holds one fully-reduced chunk. Propagate it to all others.
	wg.Add(numWorkers)
	for workerID := 0; workerID < numWorkers; workerID++ {
		go func(id int) {
			defer wg.Done()
			right := (id + 1) % numWorkers
			left := channels[id]

			// The fully-reduced chunk this worker holds after reduce-scatter
			sendChunk := (id + 1) % numWorkers

			for step := 0; step < numWorkers-1; step++ {
				time.Sleep(latency)

				dataCopy := make([]float64, len(grads[id][sendChunk]))
				copy(dataCopy, grads[id][sendChunk])
				channels[right] <- msg{chunkID: sendChunk, data: dataCopy}

				received := <-left
				copy(grads[id][received.chunkID], received.data)

				sendChunk = (sendChunk - 1 + numWorkers) % numWorkers
			}
		}(workerID)
	}
	wg.Wait()
}

func makeGradient(size int) []float64 {
	s := make([]float64, size)
	for i := range s {
		s[i] = rand.Float64()
	}
	return s
}

func sumSlice(s []float64) float64 {
	var total float64
	for _, v := range s {
		total += v
	}
	return total
}
