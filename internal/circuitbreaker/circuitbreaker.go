package circuitbreaker

import (
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

// State represents the circuit breaker state.
type State int32

const (
	StateClosed State = iota
	StateHalfOpen
	StateOpen
)

func (s State) String() string {
	switch s {
	case StateClosed:
		return "closed"
	case StateHalfOpen:
		return "half-open"
	case StateOpen:
		return "open"
	default:
		return "unknown"
	}
}

var (
	ErrCircuitOpen = errors.New("circuit breaker is open: upstream error rate exceeded 15%")
)

// Config holds circuit breaker parameters.
type Config struct {
	WindowSeconds     int           // Length of sliding window (default: 30s)
	FailureThreshold  float64       // Failure ratio threshold to trip (default: 0.15 i.e. 15%)
	MinimumRequests   int64         // Minimum requests required in window before evaluating (default: 10)
	CooldownDuration  time.Duration // Time to wait in Open state before switching to Half-Open (default: 5s)
	HalfOpenMaxProbes int32         // Maximum concurrent probe requests in Half-Open state (default: 3)
}

// DefaultConfig returns production settings matching the AetherGate spec.
func DefaultConfig() Config {
	return Config{
		WindowSeconds:     30,
		FailureThreshold:  0.15, // 15% failure ratio
		MinimumRequests:   10,
		CooldownDuration:  5 * time.Second,
		HalfOpenMaxProbes: 3,
	}
}

type timeBucket struct {
	timestamp int64 // unix second
	requests  int64
	failures  int64
	ttftSumMs int64
	ttftCount int64
}

// SlidingWindowCircuitBreaker tracks failure rates and TTFT in a lock-free sliding-window ring buffer.
type SlidingWindowCircuitBreaker struct {
	name       string
	config     Config
	state      int32 // atomic State
	lastOpened int64 // unix nano when circuit opened
	probes     int32 // active probe requests in half-open state

	buckets []timeBucket
	mu      sync.RWMutex
}

// NewCircuitBreaker initializes the sliding window circuit breaker for a provider endpoint.
func NewCircuitBreaker(name string, cfg Config) *SlidingWindowCircuitBreaker {
	if cfg.WindowSeconds <= 0 {
		cfg.WindowSeconds = 30
	}
	if cfg.FailureThreshold <= 0 {
		cfg.FailureThreshold = 0.15
	}
	if cfg.MinimumRequests <= 0 {
		cfg.MinimumRequests = 10
	}
	if cfg.CooldownDuration <= 0 {
		cfg.CooldownDuration = 5 * time.Second
	}
	if cfg.HalfOpenMaxProbes <= 0 {
		cfg.HalfOpenMaxProbes = 3
	}

	return &SlidingWindowCircuitBreaker{
		name:    name,
		config:  cfg,
		state:   int32(StateClosed),
		buckets: make([]timeBucket, cfg.WindowSeconds),
	}
}

// Name returns the provider/model endpoint name.
func (cb *SlidingWindowCircuitBreaker) Name() string {
	return cb.name
}

// State returns current circuit state.
func (cb *SlidingWindowCircuitBreaker) State() State {
	currentState := State(atomic.LoadInt32(&cb.state))
	if currentState == StateOpen {
		openedAt := atomic.LoadInt64(&cb.lastOpened)
		if time.Since(time.Unix(0, openedAt)) >= cb.config.CooldownDuration {
			// Transition to Half-Open
			if atomic.CompareAndSwapInt32(&cb.state, int32(StateOpen), int32(StateHalfOpen)) {
				atomic.StoreInt32(&cb.probes, 0)
				return StateHalfOpen
			}
		}
	}
	return currentState
}

// Allow determines if an incoming request can proceed to the upstream provider.
func (cb *SlidingWindowCircuitBreaker) Allow() error {
	state := cb.State()
	switch state {
	case StateClosed:
		return nil
	case StateHalfOpen:
		probes := atomic.AddInt32(&cb.probes, 1)
		if probes <= cb.config.HalfOpenMaxProbes {
			return nil
		}
		atomic.AddInt32(&cb.probes, -1)
		return ErrCircuitOpen
	case StateOpen:
		return ErrCircuitOpen
	default:
		return nil
	}
}

// RecordSuccess records a successful upstream call and its Time-To-First-Token in milliseconds.
func (cb *SlidingWindowCircuitBreaker) RecordSuccess(ttftMs int64) {
	cb.record(false, ttftMs)

	if State(atomic.LoadInt32(&cb.state)) == StateHalfOpen {
		// Probe succeeded: close the circuit
		atomic.StoreInt32(&cb.state, int32(StateClosed))
		atomic.StoreInt32(&cb.probes, 0)
	}
}

// RecordFailure records an upstream failure (timeout, 5xx, or 429).
func (cb *SlidingWindowCircuitBreaker) RecordFailure() {
	cb.record(true, 0)

	currentState := State(atomic.LoadInt32(&cb.state))
	if currentState == StateHalfOpen {
		// Probe failed: trip back to Open immediately
		cb.trip()
		return
	}

	// In Closed state, evaluate sliding window failure ratio
	if currentState == StateClosed {
		total, fails, _ := cb.aggregate()
		if total >= cb.config.MinimumRequests {
			ratio := float64(fails) / float64(total)
			if ratio > cb.config.FailureThreshold {
				cb.trip()
			}
		}
	}
}

func (cb *SlidingWindowCircuitBreaker) trip() {
	if atomic.CompareAndSwapInt32(&cb.state, int32(StateClosed), int32(StateOpen)) ||
		atomic.CompareAndSwapInt32(&cb.state, int32(StateHalfOpen), int32(StateOpen)) {
		atomic.StoreInt64(&cb.lastOpened, time.Now().UnixNano())
		atomic.StoreInt32(&cb.probes, 0)
	}
}

func (cb *SlidingWindowCircuitBreaker) record(failed bool, ttftMs int64) {
	nowSec := time.Now().Unix()
	idx := int(nowSec % int64(cb.config.WindowSeconds))
	if idx < 0 {
		idx = -idx
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	bucket := &cb.buckets[idx]
	if bucket.timestamp != nowSec {
		// Clear stale second bucket
		bucket.timestamp = nowSec
		bucket.requests = 0
		bucket.failures = 0
		bucket.ttftSumMs = 0
		bucket.ttftCount = 0
	}

	bucket.requests++
	if failed {
		bucket.failures++
	} else if ttftMs > 0 {
		bucket.ttftSumMs += ttftMs
		bucket.ttftCount++
	}
}

func (cb *SlidingWindowCircuitBreaker) aggregate() (totalReqs, totalFails, avgTTFT int64) {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	nowSec := time.Now().Unix()
	minValidSec := nowSec - int64(cb.config.WindowSeconds)

	var ttftSum, ttftCnt int64

	for i := 0; i < cb.config.WindowSeconds; i++ {
		b := &cb.buckets[i]
		if b.timestamp >= minValidSec && b.timestamp <= nowSec {
			totalReqs += b.requests
			totalFails += b.failures
			ttftSum += b.ttftSumMs
			ttftCnt += b.ttftCount
		}
	}

	if ttftCnt > 0 {
		avgTTFT = ttftSum / ttftCnt
	}
	return totalReqs, totalFails, avgTTFT
}

// Stats returns the aggregated sliding window metrics.
type BreakerStats struct {
	State        string  `json:"state"`
	TotalReqs    int64   `json:"total_requests"`
	TotalFails   int64   `json:"total_failures"`
	FailureRatio float64 `json:"failure_ratio"`
	AverageTTFT  int64   `json:"avg_ttft_ms"`
}

// GetStats returns the current snapshot of circuit breaker health.
func (cb *SlidingWindowCircuitBreaker) GetStats() BreakerStats {
	total, fails, avgTTFT := cb.aggregate()
	var ratio float64
	if total > 0 {
		ratio = float64(fails) / float64(total)
	}
	return BreakerStats{
		State:        cb.State().String(),
		TotalReqs:    total,
		TotalFails:   fails,
		FailureRatio: ratio,
		AverageTTFT:  avgTTFT,
	}
}
