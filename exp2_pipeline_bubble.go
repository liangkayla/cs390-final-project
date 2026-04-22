// package cs390finalproject

// Experiment 2: Pipeline Parallelism Bubble Simulation
//
// Models the GPipe-style pipeline schedule used in Megatron-LM.
// The "pipeline bubble" is idle time that occurs because:
//   - On the forward pass, stage 0 must finish before stage 1 can start
//   - On the backward pass, the last stage finishes first, earlier stages wait
//
// This simulation models a training loop where:
//   - N goroutines = N pipeline stages (each holds some layers)
//   - M microbatches flow through the pipeline per batch
//   - Each stage takes `stageLatency` to process one microbatch (forward)
//   - Backward pass takes 2x forward pass (standard assumption)
//
// The bubble fraction is:
//   bubble = (p - 1) / (p - 1 + m)
//   where p = pipeline stages, m = microbatches per batch
//
// Key insight: more microbatches → smaller bubble, but more memory usage.
// This is the fundamental tradeoff Megatron-LM must navigate.
//
// Knobs to turn:
//   --stages      number of pipeline stages
//   --microbatches number of microbatches per batch
//   --latency     ms per stage per microbatch (forward pass)
//   --sweep       if true, sweep stages 1-16 and microbatches 1-32

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sync"
	"time"
)

// StageMetrics records per-stage idle and active time.
type StageMetrics struct {
	StageID   int     `json:"stage_id"`
	ActiveMS  float64 `json:"active_ms"`
	IdleMS    float64 `json:"idle_ms"`
	BubblePct float64 `json:"bubble_pct"`
}

// RunResult holds the result of one (stages, microbatches) configuration.
type RunResult struct {
	Stages            int            `json:"stages"`
	Microbatches      int            `json:"microbatches"`
	StagLatencyMS     int            `json:"stage_latency_ms"`
	TotalWallTimeMS   float64        `json:"total_wall_time_ms"`
	TheoreticalBubble float64        `json:"theoretical_bubble_pct"` // (p-1)/(p-1+m)*100
	MeasuredBubble    float64        `json:"measured_bubble_pct"`
	StageMetrics      []StageMetrics `json:"stage_metrics"`
}

func main() {
	stages := flag.Int("stages", 0, "number of pipeline stages (0 = sweep)")
	microbatches := flag.Int("microbatches", 0, "microbatches per batch (0 = sweep)")
	latencyMS := flag.Int("latency", 5, "ms per stage per microbatch (forward)")
	sweep := flag.Bool("sweep", false, "sweep stages and microbatches")
	flag.Parse()

	var results []RunResult

	if *sweep || *stages == 0 || *microbatches == 0 {
		// Sweep meaningful combinations
		for _, p := range []int{2, 4, 8, 16} {
			for _, m := range []int{1, 2, 4, 8, 16, 32} {
				r := runPipeline(p, m, *latencyMS)
				results = append(results, r)
				fmt.Fprintf(os.Stderr,
					"stages=%2d microbatches=%2d  bubble=%.1f%% (theory=%.1f%%)  wall=%.0fms\n",
					p, m, r.MeasuredBubble, r.TheoreticalBubble, r.TotalWallTimeMS)
			}
		}
	} else {
		r := runPipeline(*stages, *microbatches, *latencyMS)
		results = append(results, r)
		fmt.Fprintf(os.Stderr,
			"stages=%d microbatches=%d  bubble=%.1f%% (theory=%.1f%%)  wall=%.0fms\n",
			r.Stages, r.Microbatches, r.MeasuredBubble, r.TheoreticalBubble, r.TotalWallTimeMS)
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	enc.Encode(results)
}

// runPipeline simulates one full training batch through the pipeline.
// Returns timing metrics including bubble fraction.
func runPipeline(numStages, numMicrobatches, stageLatencyMS int) RunResult {
	fwd := time.Duration(stageLatencyMS) * time.Millisecond
	bwd := 2 * fwd // backward pass takes 2x forward (standard assumption)

	// activationChannels[i] connects stage i -> stage i+1 (forward)
	// gradientChannels[i] connects stage i+1 -> stage i (backward)
	type token struct{ id int }
	activations := make([]chan token, numStages+1)
	gradients := make([]chan token, numStages+1)
	for i := range activations {
		activations[i] = make(chan token, numMicrobatches)
		gradients[i] = make(chan token, numMicrobatches)
	}

	// Per-stage timing tracking
	type interval struct{ start, end time.Time }
	stageActive := make([][]interval, numStages)
	for i := range stageActive {
		stageActive[i] = make([]interval, 0, numMicrobatches*2)
	}
	var mu sync.Mutex

	var wg sync.WaitGroup
	globalStart := time.Now()

	// Feed microbatches into stage 0's input
	go func() {
		for i := 0; i < numMicrobatches; i++ {
			activations[0] <- token{id: i}
		}
	}()

	// Launch each pipeline stage as a goroutine
	for stageID := 0; stageID < numStages; stageID++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			inFwd := activations[id]
			outFwd := activations[id+1] // last stage sends to [numStages] which is drained below
			inBwd := gradients[id+1]    // last stage reads from [numStages] which is fed below
			outBwd := gradients[id]

			// GPipe schedule: all forwards first, then all backwards.

			// --- Forward passes ---
			for i := 0; i < numMicrobatches; i++ {
				<-inFwd // wait for activation from upstream

				start := time.Now()
				time.Sleep(fwd)
				end := time.Now()

				mu.Lock()
				stageActive[id] = append(stageActive[id], interval{start, end})
				mu.Unlock()

				outFwd <- token{id: i} // pass to next stage (or sink for last stage)
			}

			// --- Backward passes (reverse order) ---
			for i := numMicrobatches - 1; i >= 0; i-- {
				<-inBwd // wait for gradient from downstream (or source for last stage)

				start := time.Now()
				time.Sleep(bwd)
				end := time.Now()

				mu.Lock()
				stageActive[id] = append(stageActive[id], interval{start, end})
				mu.Unlock()

				outBwd <- token{id: i} // send gradient upstream
			}
		}(stageID)
	}

	// Sink: drain activations coming out of the last stage and inject gradients back.
	// This represents the loss computation at the end of the pipeline.
	go func() {
		lastFwd := activations[numStages]
		lastBwd := gradients[numStages]
		for i := 0; i < numMicrobatches; i++ {
			t := <-lastFwd             // receive activation from last stage
			lastBwd <- token{id: t.id} // immediately send gradient back
		}
	}()

	// Drain the final gradients out of stage 0 (optimizer step happens here)
	go func() {
		for i := 0; i < numMicrobatches; i++ {
			<-gradients[0]
		}
	}()

	wg.Wait()
	totalWall := time.Since(globalStart)

	// Compute per-stage metrics
	stageMetrics := make([]StageMetrics, numStages)
	var totalBubblePct float64

	for id := 0; id < numStages; id++ {
		var activeTotal time.Duration
		for _, iv := range stageActive[id] {
			activeTotal += iv.end.Sub(iv.start)
		}
		activeMS := float64(activeTotal.Milliseconds())
		idleMS := float64(totalWall.Milliseconds()) - activeMS
		bubblePct := (idleMS / float64(totalWall.Milliseconds())) * 100

		stageMetrics[id] = StageMetrics{
			StageID:   id,
			ActiveMS:  activeMS,
			IdleMS:    idleMS,
			BubblePct: bubblePct,
		}
		totalBubblePct += bubblePct
	}

	measuredBubble := totalBubblePct / float64(numStages)

	// Theoretical bubble fraction: (p-1) / (p-1+m)
	p := float64(numStages)
	m := float64(numMicrobatches)
	theoreticalBubble := ((p - 1) / (p - 1 + m)) * 100

	return RunResult{
		Stages:            numStages,
		Microbatches:      numMicrobatches,
		StagLatencyMS:     stageLatencyMS,
		TotalWallTimeMS:   float64(totalWall.Milliseconds()),
		TheoreticalBubble: theoreticalBubble,
		MeasuredBubble:    measuredBubble,
		StageMetrics:      stageMetrics,
	}
}
