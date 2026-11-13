package circuitbreaker

import (
	"testing"
	"time"
)

func TestSlidingWindowCircuitBreaker(t *testing.T) {
	cfg := Config{
		WindowSeconds:     5,
		FailureThreshold:  0.15,
		MinimumRequests:   10,
		CooldownDuration:  50 * time.Millisecond,
		HalfOpenMaxProbes: 2,
	}

	cb := NewCircuitBreaker("test-provider", cfg)

	if cb.State() != StateClosed {
		t.Fatalf("expected initial state to be Closed, got %v", cb.State())
	}
	if err := cb.Allow(); err != nil {
		t.Fatalf("expected Allow() to succeed on Closed state, got %v", err)
	}

	// 1. Record 8 successes, 0 failures (0% failure ratio)
	for i := 0; i < 8; i++ {
		cb.RecordSuccess(100)
	}
	if cb.State() != StateClosed {
		t.Fatalf("expected state to stay Closed, got %v", cb.State())
	}

	// 2. Record 2 failures (2 out of 10 = 20% failure ratio > 15% threshold)
	cb.RecordFailure()
	cb.RecordFailure()

	if cb.State() != StateOpen {
		t.Fatalf("expected circuit to trip to Open after 20%% failures, got %v (stats: %+v)", cb.State(), cb.GetStats())
	}

	// Allow() should now fail fast
	if err := cb.Allow(); err != ErrCircuitOpen {
		t.Fatalf("expected ErrCircuitOpen, got %v", err)
	}

	// 3. Wait for cooldown duration (50ms)
	time.Sleep(70 * time.Millisecond)

	// State should now be HalfOpen
	if cb.State() != StateHalfOpen {
		t.Fatalf("expected state to transition to HalfOpen, got %v", cb.State())
	}

	// In HalfOpen, allow up to HalfOpenMaxProbes (2)
	if err := cb.Allow(); err != nil {
		t.Fatalf("first probe should be allowed, got %v", err)
	}
	if err := cb.Allow(); err != nil {
		t.Fatalf("second probe should be allowed, got %v", err)
	}
	// Third probe should be rejected
	if err := cb.Allow(); err != ErrCircuitOpen {
		t.Fatalf("expected third probe to be rejected in HalfOpen, got %v", err)
	}

	// Record success for probe: circuit should heal and return to Closed
	cb.RecordSuccess(80)
	if cb.State() != StateClosed {
		t.Fatalf("expected successful probe to close circuit, got %v", cb.State())
	}
}
