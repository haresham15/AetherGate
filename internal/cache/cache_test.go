package cache

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/aethergate/aethergate/internal/embedding"
	"github.com/aethergate/aethergate/internal/models"
)

func TestEscapeRediSearchTag(t *testing.T) {
	// Attempt query injection breakout payloads
	malicious := "evil_tenant} | @tenant_id:{*"
	escaped := EscapeRediSearchTag(malicious)

	expected := `evil_tenant\}\ \|\ \@tenant_id\:\{\*`
	if escaped != expected {
		t.Fatalf("expected escaped tag %q, got %q", expected, escaped)
	}

	// Normal alphanumeric
	normal := "tenant-prod_123"
	escapedNormal := EscapeRediSearchTag(normal)
	expectedNormal := `tenant\-prod_123`
	if escapedNormal != expectedNormal {
		t.Fatalf("expected escaped tag %q, got %q", expectedNormal, escapedNormal)
	}
}

func TestFloat32BytesConversion(t *testing.T) {
	vec := []float32{0.1234, -0.5678, 0.9999, 0.0, -1.0}
	bytes := Float32ToBytes(vec)
	if len(bytes) != len(vec)*4 {
		t.Fatalf("expected %d bytes, got %d", len(vec)*4, len(bytes))
	}

	restored := BytesToFloat32(bytes)
	if len(restored) != len(vec) {
		t.Fatalf("expected %d floats, got %d", len(vec), len(restored))
	}

	for i := range vec {
		if math.Abs(float64(vec[i]-restored[i])) > 1e-6 {
			t.Fatalf("mismatch at index %d: expected %f, got %f", i, vec[i], restored[i])
		}
	}
}

func TestMemorySemanticCache(t *testing.T) {
	cache := NewMemorySemanticCache(0.08, 1*time.Hour)
	ctx := context.Background()

	embedder := embedding.NewFastMiniLMEmbedder()
	prompt := "How do I implement mutual TLS in Go?"
	vec, err := embedder.Embed(ctx, prompt)
	if err != nil {
		t.Fatalf("embed failed: %v", err)
	}

	tenant1 := "tenant-alpha"
	sysHash := HashSystemPrompt("You are an expert Go developer.")
	tier := 1
	expectedAnswer := "To implement mutual TLS in Go, configure tls.Config with ClientAuth: tls.RequireAndVerifyClientCert."

	// 1. Initial lookup must miss
	entry, hit, err := cache.Lookup(ctx, tenant1, sysHash, tier, vec)
	if err != nil {
		t.Fatalf("lookup error: %v", err)
	}
	if hit || entry != nil {
		t.Fatalf("expected cache miss on empty cache")
	}

	// 2. Store response
	if err := cache.Store(ctx, tenant1, sysHash, tier, vec, expectedAnswer); err != nil {
		t.Fatalf("store error: %v", err)
	}

	// 3. Exact lookup must hit
	entry, hit, err = cache.Lookup(ctx, tenant1, sysHash, tier, vec)
	if err != nil || !hit || entry == nil {
		t.Fatalf("expected cache hit, got hit=%v, err=%v", hit, err)
	}
	if entry.Response != expectedAnswer {
		t.Fatalf("expected %q, got %q", expectedAnswer, entry.Response)
	}
	if entry.Distance > 0.08 {
		t.Fatalf("expected distance <= 0.08, got %f", entry.Distance)
	}

	// 4. Multi-tenant tag isolation: different tenant must miss
	entry2, hit2, _ := cache.Lookup(ctx, "tenant-beta", sysHash, tier, vec)
	if hit2 || entry2 != nil {
		t.Fatalf("expected cache miss across tenants for isolation")
	}

	// 5. Persona isolation: different system prompt must miss
	differentSysHash := HashSystemPrompt("You are a casual assistant.")
	entry3, hit3, _ := cache.Lookup(ctx, tenant1, differentSysHash, tier, vec)
	if hit3 || entry3 != nil {
		t.Fatalf("expected cache miss across system prompt personas")
	}

	// 6. Model tier isolation: different tier must miss
	entry4, hit4, _ := cache.Lookup(ctx, tenant1, sysHash, 3, vec)
	if hit4 || entry4 != nil {
		t.Fatalf("expected cache miss across model tiers")
	}
}

func TestReplayStreaming(t *testing.T) {
	cache := NewMemorySemanticCache(0.08, 1*time.Hour)
	ctx := context.Background()

	cachedText := "AetherGate delivers sub-10ms Redis HNSW caching and SSE failover."
	entry := &CacheEntry{
		Response: cachedText,
	}

	outChan := make(chan models.StreamChunk, 20)
	errChan := make(chan error, 1)

	go cache.ReplayStreaming(ctx, entry, "gpt-4o", outChan, errChan)

	var reassembled string
	var chunkCount int
	var receivedStop bool

	for chunk := range outChan {
		chunkCount++
		for _, choice := range chunk.Choices {
			reassembled += choice.Delta.Content
			if choice.FinishReason == "stop" {
				receivedStop = true
			}
		}
	}

	if !receivedStop {
		t.Fatalf("expected terminating chunk with finish_reason: stop")
	}
	if reassembled != cachedText {
		t.Fatalf("expected reassembled text %q, got %q", cachedText, reassembled)
	}
	if chunkCount < 3 {
		t.Fatalf("expected multiple micro-batch chunks, got %d", chunkCount)
	}
}
