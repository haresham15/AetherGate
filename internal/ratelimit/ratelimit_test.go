package ratelimit

import (
	"context"
	"testing"
)

func TestMemoryDualRateLimiter(t *testing.T) {
	limiter := NewMemoryDualRateLimiter()
	ctx := context.Background()
	tenant := "tenant-test-123"

	rpm := 10.0
	tpm := 100.0

	// 1. First request should be allowed (1 req, 20 tokens)
	allowed, err := limiter.Allow(ctx, tenant, 20, rpm, tpm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !allowed {
		t.Fatalf("expected request to be allowed")
	}

	// 2. Consume up to token capacity
	allowed, err = limiter.Allow(ctx, tenant, 70, rpm, tpm)
	if err != nil || !allowed {
		t.Fatalf("expected request with 70 tokens to be allowed, got allowed=%v err=%v", allowed, err)
	}

	// Now remaining tokens = 100 - 20 - 70 = 10 tokens
	// Requesting 15 tokens should be rejected because 15 > 10
	allowed, err = limiter.Allow(ctx, tenant, 15, rpm, tpm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if allowed {
		t.Fatalf("expected request exceeding TPM to be rejected")
	}

	// 3. Reset and verify allowance restored
	if err := limiter.Reset(ctx, tenant); err != nil {
		t.Fatalf("failed to reset limiter: %v", err)
	}

	allowed, err = limiter.Allow(ctx, tenant, 50, rpm, tpm)
	if err != nil || !allowed {
		t.Fatalf("expected request to be allowed after reset")
	}
}

func TestMemoryDualRateLimiterRPMExhaustion(t *testing.T) {
	limiter := NewMemoryDualRateLimiter()
	ctx := context.Background()
	tenant := "tenant-rpm-test"

	rpm := 3.0
	tpm := 10000.0

	// 3 requests allowed
	for i := 0; i < 3; i++ {
		allowed, err := limiter.Allow(ctx, tenant, 10, rpm, tpm)
		if err != nil || !allowed {
			t.Fatalf("iteration %d: expected request to be allowed", i)
		}
	}

	// 4th request must be rejected on RPM
	allowed, err := limiter.Allow(ctx, tenant, 10, rpm, tpm)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if allowed {
		t.Fatalf("expected 4th request to be rejected due to RPM limit")
	}
}
