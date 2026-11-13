package embedding

import (
	"context"
	"math"
	"sync"
	"testing"
	"time"
)

func TestFastMiniLMEmbedder(t *testing.T) {
	embedder := NewFastMiniLMEmbedder()
	ctx := context.Background()

	text := "What is the capital of France?"
	vec, err := embedder.Embed(ctx, text)
	if err != nil {
		t.Fatalf("Embed failed: %v", err)
	}

	if len(vec) != 384 {
		t.Fatalf("expected 384 dimensions, got %d", len(vec))
	}

	// Verify L2 norm is ~1.0
	var sumSq float64
	for _, v := range vec {
		sumSq += float64(v * v)
	}
	norm := math.Sqrt(sumSq)
	if math.Abs(norm-1.0) > 1e-4 {
		t.Fatalf("expected unit norm ~1.0, got %f", norm)
	}
}

func TestEmbedderSemanticDistance(t *testing.T) {
	embedder := NewFastMiniLMEmbedder()
	ctx := context.Background()

	t1 := "What is the capital of France?"
	t2 := "What is the capital of France?" // Identical
	t3 := "Tell me the capital city of France" // Semantic paraphrase
	t4 := "How do I configure a Kubernetes ingress controller in AWS EKS?" // Completely different

	v1, _ := embedder.Embed(ctx, t1)
	v2, _ := embedder.Embed(ctx, t2)
	v3, _ := embedder.Embed(ctx, t3)
	v4, _ := embedder.Embed(ctx, t4)

	distIdentical, _ := CosineDistance(v1, v2)
	if distIdentical > 1e-5 {
		t.Fatalf("identical text should have cosine distance ~0, got %f", distIdentical)
	}

	distParaphrase, _ := CosineDistance(v1, v3)
	distUnrelated, _ := CosineDistance(v1, v4)

	t.Logf("Distance identical: %f", distIdentical)
	t.Logf("Distance paraphrase: %f", distParaphrase)
	t.Logf("Distance unrelated: %f", distUnrelated)

	if distParaphrase >= distUnrelated {
		t.Fatalf("expected paraphrase distance (%f) to be lower than unrelated distance (%f)", distParaphrase, distUnrelated)
	}
}

func TestWorkerPoolConcurrency(t *testing.T) {
	embedder := NewFastMiniLMEmbedder()
	pool := NewWorkerPool(embedder, 4, 32)
	defer pool.Close()

	ctx := context.Background()
	concurrency := 20
	var wg sync.WaitGroup
	wg.Add(concurrency)

	start := time.Now()
	for i := 0; i < concurrency; i++ {
		go func(idx int) {
			defer wg.Done()
			vec, err := pool.Embed(ctx, "High throughput vector embedding benchmark test")
			if err != nil {
				t.Errorf("worker %d failed: %v", idx, err)
			}
			if len(vec) != 384 {
				t.Errorf("worker %d got invalid dims: %d", idx, len(vec))
			}
		}(i)
	}
	wg.Wait()
	elapsed := time.Since(start)
	t.Logf("Embedded %d requests across worker pool in %v (avg %v per req)", concurrency, elapsed, elapsed/time.Duration(concurrency))
}
