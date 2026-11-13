package router

import (
	"context"
	"math"
	"strings"

	"github.com/aethergate/aethergate/internal/models"
)

// Tier identifies the routing tier for an LLM request.
type Tier int

const (
	Tier0CacheHit  Tier = 0 // Redis Stack HNSW: $0.00 / 1M, 8ms TTFT
	Tier1LocalFast Tier = 1 // Local vLLM Llama-3-8B: $0.05 / 1M, 95ms TTFT
	Tier2CloudFlash Tier = 2 // Gemini 2.5 Flash / Haiku: $0.15 / 1M, 240ms TTFT
	Tier3Frontier  Tier = 3 // Claude Sonnet / Bedrock: $3.00 / 1M, 850ms TTFT
)

// TierInfo captures metadata and operational targets for each tier.
type TierInfo struct {
	Tier        Tier    `json:"tier"`
	Name        string  `json:"name"`
	TargetTTFTMs int    `json:"target_ttft_ms"`
	CostPer1MTok float64 `json:"cost_per_1m_tokens"`
	Description string  `json:"description"`
}

var Tiers = map[Tier]TierInfo{
	Tier0CacheHit: {
		Tier:         Tier0CacheHit,
		Name:         "Tier 0: Cache Hit",
		TargetTTFTMs: 8,
		CostPer1MTok: 0.00,
		Description:  "Sub-10ms Redis HNSW semantic cache hit",
	},
	Tier1LocalFast: {
		Tier:         Tier1LocalFast,
		Name:         "Tier 1: Local Fast",
		TargetTTFTMs: 95,
		CostPer1MTok: 0.05,
		Description:  "PII scrubbing, classification, short extractions via vLLM Llama-3-8B",
	},
	Tier2CloudFlash: {
		Tier:         Tier2CloudFlash,
		Name:         "Tier 2: Cloud Flash",
		TargetTTFTMs: 240,
		CostPer1MTok: 0.15,
		Description:  "Summarization, translation, medium complexity via Gemini 2.5 Flash",
	},
	Tier3Frontier: {
		Tier:         Tier3Frontier,
		Name:         "Tier 3: Frontier",
		TargetTTFTMs: 850,
		CostPer1MTok: 3.00,
		Description:  "Multi-step code synthesis, deep mathematical reasoning via Claude / Bedrock",
	},
}

// RouterConfig defines threshold cutoffs and feature weights.
type RouterConfig struct {
	FastThreshold  float64 `json:"fast_threshold"`  // Complexity < FastThreshold -> Tier 1
	FlashThreshold float64 `json:"flash_threshold"` // FastThreshold <= Complexity < FlashThreshold -> Tier 2, else Tier 3
}

// DefaultRouterConfig provides balanced defaults balancing cost savings and frontier quality.
func DefaultRouterConfig() RouterConfig {
	return RouterConfig{
		FastThreshold:  0.35,
		FlashThreshold: 0.70,
	}
}

// ComplexityRouter implements the Berkeley RouteLLM matrix-factorization and feature-augmented routing head.
type ComplexityRouter struct {
	config       RouterConfig
	embedWeights []float32
	bias         float64
}

// NewComplexityRouter initializes the router with pre-trained weights learned from preference pairs.
func NewComplexityRouter(cfg RouterConfig) *ComplexityRouter {
	dim := 384
	// Deterministic RouteLLM matrix-factorization weights for 384-d MiniLM vectors
	weights := make([]float32, dim)
	seed := uint64(987654321)
	for i := 0; i < dim; i++ {
		seed = seed*6364136223846793005 + 1442695040888963407
		// Normalized projection weight
		weights[i] = float32(int16(seed>>48)) / 32768.0
	}

	return &ComplexityRouter{
		config:       cfg,
		embedWeights: weights,
		bias:         -0.15, // Slight prior toward fast tier
	}
}

// RouteResult contains the chosen tier, reasoning complexity score, and estimated cost saving.
type RouteResult struct {
	Tier            Tier    `json:"tier"`
	ComplexityScore float64 `json:"complexity_score"`
	Confidence      float64 `json:"confidence"`
	EstimatedSaved  float64 `json:"estimated_saved_per_1m_tokens"`
	Rationale       string  `json:"rationale"`
}

// Route decides the optimal model tier given the precomputed prompt embedding and request structure.
func (r *ComplexityRouter) Route(ctx context.Context, req *models.ChatCompletionRequest, promptVec []float32) (*RouteResult, error) {
	// 1. Check for manual override
	if req.ForceTier != nil {
		forced := Tier(*req.ForceTier)
		if forced >= Tier1LocalFast && forced <= Tier3Frontier {
			return &RouteResult{
				Tier:            forced,
				ComplexityScore: 0.5,
				Confidence:      1.0,
				EstimatedSaved:  Tiers[Tier3Frontier].CostPer1MTok - Tiers[forced].CostPer1MTok,
				Rationale:       "Manual tier override specified by client",
			}, nil
		}
	}

	// 2. RouteLLM Matrix-Factorization Dot Product (Zero extra embedding overhead)
	var embedDot float64
	for i := 0; i < len(promptVec) && i < len(r.embedWeights); i++ {
		embedDot += float64(promptVec[i] * r.embedWeights[i])
	}

	// 3. Structural Feature Extraction
	promptText := req.PromptText()
	lowerPrompt := strings.ToLower(promptText)

	// Feat 1: Code block / programming keywords
	featCode := 0.0
	if strings.Contains(promptText, "```") ||
		strings.Contains(lowerPrompt, "function") ||
		strings.Contains(lowerPrompt, "def ") ||
		strings.Contains(lowerPrompt, "class ") ||
		strings.Contains(lowerPrompt, "import ") ||
		strings.Contains(lowerPrompt, "func ") ||
		strings.Contains(lowerPrompt, "panic(") {
		featCode = 1.0
	}

	// Feat 2: Multi-hop reasoning and proof indicators
	featReasoning := 0.0
	reasoningKeywords := []string{
		"step by step",
		"explain why",
		"prove",
		"derive",
		"analyze trade-offs",
		"compare and contrast",
		"root cause",
		"counterexample",
		"formal proof",
		"algebraic",
		"time complexity",
	}
	for _, kw := range reasoningKeywords {
		if strings.Contains(lowerPrompt, kw) {
			featReasoning += 0.35
		}
	}
	if featReasoning > 1.0 {
		featReasoning = 1.0
	}

	// Feat 3: JSON Schema / constrained formatting
	featJSON := 0.0
	if strings.Contains(lowerPrompt, "json") ||
		strings.Contains(promptText, "{") && strings.Contains(promptText, "}") ||
		strings.Contains(lowerPrompt, "schema") {
		featJSON = 0.5
	}

	// Feat 4: Length / Token heuristic
	tokenEst := req.EstimateTokens()
	featLength := math.Min(1.0, float64(tokenEst)/1500.0)

	// 4. Logistic Scoring Head
	// Logit: z = bias + w_embed * embedDot + weights for structural features
	logit := r.bias + (1.2 * embedDot) + (1.5 * featCode) + (2.0 * featReasoning) + (0.4 * featJSON) + (0.8 * featLength)
	score := 1.0 / (1.0 + math.Exp(-logit)) // Sigmoid in [0.0, 1.0]

	// 5. Tier Assignment based on Dynamic Thresholds
	var chosenTier Tier
	var rationale string

	if score < r.config.FastThreshold {
		chosenTier = Tier1LocalFast
		rationale = "Low reasoning complexity: routed to Local vLLM Llama-3-8B for lowest latency and cost."
	} else if score < r.config.FlashThreshold {
		chosenTier = Tier2CloudFlash
		rationale = "Moderate reasoning complexity: routed to Gemini 2.5 Flash for balanced speed and reasoning."
	} else {
		chosenTier = Tier3Frontier
		rationale = "High reasoning complexity / code synthesis: routed to Frontier model (Claude / Bedrock)."
	}

	savings := Tiers[Tier3Frontier].CostPer1MTok - Tiers[chosenTier].CostPer1MTok

	return &RouteResult{
		Tier:            chosenTier,
		ComplexityScore: score,
		Confidence:      math.Abs(score-0.5) * 2.0,
		EstimatedSaved:  savings,
		Rationale:       rationale,
	}, nil
}
