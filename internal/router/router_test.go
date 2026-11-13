package router

import (
	"context"
	"testing"

	"github.com/aethergate/aethergate/internal/embedding"
	"github.com/aethergate/aethergate/internal/models"
)

func TestComplexityRouter(t *testing.T) {
	router := NewComplexityRouter(DefaultRouterConfig())
	embedder := embedding.NewFastMiniLMEmbedder()
	ctx := context.Background()

	// 1. Simple classification prompt
	simpleReq := &models.ChatCompletionRequest{
		Model: "gpt-4o",
		Messages: []models.ChatMessage{
			{Role: models.RoleUser, Content: "Classify this email as spam or not spam: You won a lottery!"},
		},
	}
	simpleVec, _ := embedder.Embed(ctx, simpleReq.PromptText())
	resSimple, err := router.Route(ctx, simpleReq, simpleVec)
	if err != nil {
		t.Fatalf("routing error: %v", err)
	}

	t.Logf("Simple prompt score: %f, Tier: %v", resSimple.ComplexityScore, resSimple.Tier)
	if resSimple.Tier == Tier3Frontier {
		t.Fatalf("expected simple prompt to be routed to fast/flash tier, got Tier 3 Frontier")
	}

	// 2. High complexity code + proof prompt
	complexReq := &models.ChatCompletionRequest{
		Model: "gpt-4o",
		Messages: []models.ChatMessage{
			{
				Role: models.RoleUser,
				Content: "```go\nfunc Dijkstra(g Graph) {}\n```\nExplain step by step how to prove the formal correctness and time complexity of Dijkstra's algorithm. Derive the algebraic recurrence and analyze trade-offs against Bellman-Ford.",
			},
		},
	}
	complexVec, _ := embedder.Embed(ctx, complexReq.PromptText())
	resComplex, err := router.Route(ctx, complexReq, complexVec)
	if err != nil {
		t.Fatalf("routing error: %v", err)
	}

	t.Logf("Complex prompt score: %f, Tier: %v", resComplex.ComplexityScore, resComplex.Tier)
	if resComplex.Tier != Tier3Frontier {
		t.Fatalf("expected complex code/proof prompt to be routed to Tier 3 Frontier, got %v", resComplex.Tier)
	}
	if resComplex.EstimatedSaved != 0.0 {
		t.Fatalf("expected 0 savings when using Frontier tier")
	}

	// 3. Forced tier override
	forceTier := int(Tier1LocalFast)
	forcedReq := &models.ChatCompletionRequest{
		Model:     "gpt-4o",
		ForceTier: &forceTier,
		Messages: []models.ChatMessage{
			{Role: models.RoleUser, Content: "Write an operating system kernel"},
		},
	}
	resForced, err := router.Route(ctx, forcedReq, complexVec)
	if err != nil {
		t.Fatalf("routing error: %v", err)
	}
	if resForced.Tier != Tier1LocalFast {
		t.Fatalf("expected forced tier to be Tier 1, got %v", resForced.Tier)
	}
	if resForced.EstimatedSaved <= 2.50 {
		t.Fatalf("expected significant savings for Tier 1 override, got %f", resForced.EstimatedSaved)
	}
}
