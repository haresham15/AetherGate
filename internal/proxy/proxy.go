package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/aethergate/aethergate/internal/cache"
	"github.com/aethergate/aethergate/internal/circuitbreaker"
	"github.com/aethergate/aethergate/internal/embedding"
	"github.com/aethergate/aethergate/internal/models"
	"github.com/aethergate/aethergate/internal/ratelimit"
	"github.com/aethergate/aethergate/internal/router"
	"github.com/aethergate/aethergate/internal/telemetry"
)

const (
	FirstByteDeadline   = 1500 * time.Millisecond
	DefaultTenantRPM    = 600.0                // 600 requests per minute
	DefaultTenantTPM    = 500000.0             // 500k tokens per minute
	MaxRequestBodyBytes = 10 * 1024 * 1024     // 10 MB maximum request payload
)

// GatewayConfig holds configuration for the AetherGate reverse proxy.
type GatewayConfig struct {
	DefaultRPM float64
	DefaultTPM float64
	AdminToken string
}

// ResilientProxy coordinates the 5-stage pipeline, provider failover, and semantic cache.
type ResilientProxy struct {
	cfg         GatewayConfig
	embedder    *embedding.WorkerPool
	cache       cache.SemanticCacheEngine
	router      *router.ComplexityRouter
	limiter     ratelimit.DualRateLimiter
	metrics     *telemetry.Metrics
	breakers    map[string]*circuitbreaker.SlidingWindowCircuitBreaker
	providers   map[router.Tier][]Provider
	mu          sync.RWMutex
}

// NewResilientProxy initializes the gateway proxy engine.
func NewResilientProxy(
	cfg GatewayConfig,
	embedder *embedding.WorkerPool,
	c cache.SemanticCacheEngine,
	r *router.ComplexityRouter,
	limiter ratelimit.DualRateLimiter,
	m *telemetry.Metrics,
) *ResilientProxy {
	if cfg.DefaultRPM <= 0 {
		cfg.DefaultRPM = DefaultTenantRPM
	}
	if cfg.DefaultTPM <= 0 {
		cfg.DefaultTPM = DefaultTenantTPM
	}

	return &ResilientProxy{
		cfg:       cfg,
		embedder:  embedder,
		cache:     c,
		router:    r,
		limiter:   limiter,
		metrics:   m,
		breakers:  make(map[string]*circuitbreaker.SlidingWindowCircuitBreaker),
		providers: make(map[router.Tier][]Provider),
	}
}

// RegisterProvider attaches a provider and its circuit breaker to a specific routing tier.
func (p *ResilientProxy) RegisterProvider(provider Provider) {
	p.mu.Lock()
	defer p.mu.Unlock()

	tier := provider.Tier()
	p.providers[tier] = append(p.providers[tier], provider)

	name := provider.Name()
	if _, exists := p.breakers[name]; !exists {
		p.breakers[name] = circuitbreaker.NewCircuitBreaker(name, circuitbreaker.DefaultConfig())
	}
}

// GetCircuitBreaker returns the breaker for a provider name.
func (p *ResilientProxy) GetCircuitBreaker(name string) *circuitbreaker.SlidingWindowCircuitBreaker {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.breakers[name]
}

// ServeHTTP handles OpenAI-compatible /v1/chat/completions requests.
func (p *ResilientProxy) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	start := time.Now()

	// ------------------------------------------------------------------------
	// Stage 1: Ingress & Tenant Rate Limiting
	// ------------------------------------------------------------------------
	tenantID := p.extractTenant(r)

	// Guard against memory exhaustion DoS by bounding request body to MaxRequestBodyBytes (10MB)
	r.Body = http.MaxBytesReader(w, r.Body, MaxRequestBodyBytes)

	// Decode request body
	var req models.ChatCompletionRequest
	bodyBytes, err := io.ReadAll(r.Body)
	if err != nil {
		if strings.Contains(err.Error(), "request body too large") {
			http.Error(w, "Request body exceeds 10MB payload limit", http.StatusRequestEntityTooLarge)
			return
		}
		http.Error(w, "Failed to read request body: "+err.Error(), http.StatusBadRequest)
		return
	}
	if err := json.Unmarshal(bodyBytes, &req); err != nil {
		http.Error(w, "Invalid JSON payload: "+err.Error(), http.StatusBadRequest)
		return
	}

	estTokens := req.EstimateTokens()
	allowed, err := p.limiter.Allow(r.Context(), tenantID, estTokens, p.cfg.DefaultRPM, p.cfg.DefaultTPM)
	if err != nil || !allowed {
		p.metrics.RateLimitRejectionsTotal.WithLabelValues(tenantID, "rpm_tpm").Inc()
		p.metrics.RequestsTotal.WithLabelValues(tenantID, "unknown", "429").Inc()
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"error": map[string]string{
				"message": "AetherGate Rate Limit Exceeded: RPM or TPM threshold breached for tenant",
				"type":    "rate_limit_error",
				"param":   "rate_limit",
				"code":    "rate_limit_exceeded",
			},
		})
		return
	}

	promptText := req.PromptText()
	if promptText == "" {
		http.Error(w, "Empty prompt messages", http.StatusBadRequest)
		return
	}

	// ------------------------------------------------------------------------
	// Stage 2: In-Process ONNX Embedding
	// ------------------------------------------------------------------------
	embedStart := time.Now()
	vec, err := p.embedder.Embed(r.Context(), promptText)
	embedDuration := time.Since(embedStart).Seconds()
	p.metrics.EmbeddingDuration.Observe(embedDuration)
	if err != nil {
		http.Error(w, "Embedding generation failed: "+err.Error(), http.StatusInternalServerError)
		return
	}

	// ------------------------------------------------------------------------
	// Stage 3: Redis HNSW Semantic Cache Lookup
	// ------------------------------------------------------------------------
	systemHash := cache.HashSystemPrompt(req.SystemPrompt())
	approxTier := 0
	if req.ForceTier != nil {
		approxTier = *req.ForceTier
	}

	if !req.SkipCache {
		entry, hit, _ := p.cache.Lookup(r.Context(), tenantID, systemHash, approxTier, vec)
		if hit && entry != nil {
			p.metrics.CacheHitsTotal.WithLabelValues(tenantID).Inc()
			p.metrics.RequestsTotal.WithLabelValues(tenantID, "tier0_cache", "200").Inc()
			p.metrics.DollarsSavedTotal.WithLabelValues(tenantID).Add(router.Tiers[router.Tier3Frontier].CostPer1MTok)
			p.metrics.RequestDuration.WithLabelValues("tier0_cache", "true").Observe(time.Since(start).Seconds())
			p.metrics.TTFTHistogram.WithLabelValues("redis_hnsw_cache", "tier0").Observe(time.Since(start).Seconds())

			if req.Stream {
				// Replay streaming simulation micro-batches over SSE
				w.Header().Set("Content-Type", "text/event-stream")
				w.Header().Set("Cache-Control", "no-cache")
				w.Header().Set("Connection", "keep-alive")
				w.Header().Set("X-Accel-Buffering", "no")
				w.Header().Set("X-AetherGate-Cache", "HIT")
				w.WriteHeader(http.StatusOK)

				flusher, ok := w.(http.Flusher)
				if ok {
					flusher.Flush()
				}

				outChan := make(chan models.StreamChunk, 32)
				errChan := make(chan error, 1)
				go p.cache.ReplayStreaming(r.Context(), entry, req.Model, outChan, errChan)

				for chunk := range outChan {
					w.Write(models.FormatSSEChunk(&chunk))
					if ok {
						flusher.Flush()
					}
				}
				w.Write(models.DoneSSEChunk())
				if ok {
					flusher.Flush()
				}
				return
			}

			// Non-streaming cache response
			resp := models.NewChatCompletionResponse(req.Model, entry.Response, "stop", estTokens, len(entry.Response)/4)
			resp.Cached = true
			resp.RoutingTier = int(router.Tier0CacheHit)
			resp.SelectedProvider = "redis_hnsw_cache"

			w.Header().Set("Content-Type", "application/json")
			w.Header().Set("X-AetherGate-Cache", "HIT")
			json.NewEncoder(w).Encode(resp)
			return
		}
	}

	p.metrics.CacheMissesTotal.WithLabelValues(tenantID).Inc()

	// ------------------------------------------------------------------------
	// Stage 4: Local Complexity Classification
	// ------------------------------------------------------------------------
	routeRes, err := p.router.Route(r.Context(), &req, vec)
	if err != nil {
		routeRes = &router.RouteResult{Tier: router.Tier1LocalFast}
	}

	tier := routeRes.Tier
	tierStr := fmt.Sprintf("tier%d", tier)
	p.metrics.DollarsSavedTotal.WithLabelValues(tenantID).Add(routeRes.EstimatedSaved)

	// ------------------------------------------------------------------------
	// Stage 5: Resilient SSE Proxy & Async Write-Back with First-Byte Failover
	// ------------------------------------------------------------------------
	primary, secondary := p.selectProviders(tier)
	if primary == nil {
		http.Error(w, "No healthy upstream providers available for tier "+tierStr, http.StatusServiceUnavailable)
		return
	}

	if req.Stream {
		p.handleStreamingProxy(w, r, &req, tenantID, systemHash, int(tier), vec, primary, secondary, start)
	} else {
		p.handleNonStreamingProxy(w, r, &req, tenantID, systemHash, int(tier), vec, primary, secondary, start)
	}
}

// selectProviders selects the primary provider and a suitable secondary failover candidate.
func (p *ResilientProxy) selectProviders(tier router.Tier) (Provider, Provider) {
	p.mu.RLock()
	defer p.mu.RUnlock()

	tierProviders := p.providers[tier]
	var healthy []Provider

	for _, prov := range tierProviders {
		cb := p.breakers[prov.Name()]
		if cb == nil || cb.Allow() == nil {
			healthy = append(healthy, prov)
		}
	}

	if len(healthy) == 0 {
		// Fallback to Tier 2 or Tier 3 if current tier is exhausted
		for _, altTier := range []router.Tier{router.Tier2CloudFlash, router.Tier3Frontier, router.Tier1LocalFast} {
			if altTier != tier {
				for _, prov := range p.providers[altTier] {
					cb := p.breakers[prov.Name()]
					if cb == nil || cb.Allow() == nil {
						healthy = append(healthy, prov)
					}
				}
			}
		}
	}

	if len(healthy) == 0 {
		return nil, nil
	}

	primary := healthy[0]
	var secondary Provider
	if len(healthy) > 1 {
		secondary = healthy[1]
	}
	return primary, secondary
}

// handleStreamingProxy executes first-byte deadline failover and streams normalized SSE frames.
func (p *ResilientProxy) handleStreamingProxy(
	w http.ResponseWriter,
	r *http.Request,
	req *models.ChatCompletionRequest,
	tenantID, systemHash string,
	modelTier int,
	vec []float32,
	primary, secondary Provider,
	overallStart time.Time,
) {
	flusher, hasFlusher := w.(http.Flusher)

	primaryCtx, cancelPrimary := context.WithCancel(r.Context())
	defer cancelPrimary()

	streamChan, errChan := primary.CompleteStream(primaryCtx, req)
	cbPrimary := p.GetCircuitBreaker(primary.Name())

	var firstChunk *models.StreamChunk
	var activeProvider = primary

	// 1. First-Byte Deadline: Wait up to 1,500ms for first token chunk
	select {
	case <-time.After(FirstByteDeadline):
		// Deadline exceeded: trigger transparent failover
		if cbPrimary != nil {
			cbPrimary.RecordFailure()
		}
		cancelPrimary()
		p.metrics.FailoverEventsTotal.WithLabelValues(primary.Name(), secondaryName(secondary), "first_byte_timeout").Inc()

		if secondary != nil {
			activeProvider = secondary
			streamChan, errChan = secondary.CompleteStream(r.Context(), req)
		} else {
			http.Error(w, "Upstream timeout: primary exceeded 1,500ms first-byte deadline and no secondary available", http.StatusGatewayTimeout)
			return
		}

	case err := <-errChan:
		if err != nil {
			if cbPrimary != nil {
				cbPrimary.RecordFailure()
			}
			cancelPrimary()
			p.metrics.FailoverEventsTotal.WithLabelValues(primary.Name(), secondaryName(secondary), "upstream_error").Inc()

			if secondary != nil {
				activeProvider = secondary
				streamChan, errChan = secondary.CompleteStream(r.Context(), req)
			} else {
				http.Error(w, "Upstream provider failed: "+err.Error(), http.StatusBadGateway)
				return
			}
		}

	case chunk, ok := <-streamChan:
		if !ok {
			// Channel closed with no chunks
			if secondary != nil {
				activeProvider = secondary
				streamChan, errChan = secondary.CompleteStream(r.Context(), req)
			} else {
				http.Error(w, "Upstream closed without tokens", http.StatusBadGateway)
				return
			}
		} else {
			firstChunk = &chunk
		}
	}

	ttft := time.Since(overallStart)
	p.metrics.TTFTHistogram.WithLabelValues(activeProvider.Name(), fmt.Sprintf("tier%d", modelTier)).Observe(ttft.Seconds())

	cbActive := p.GetCircuitBreaker(activeProvider.Name())
	if cbActive != nil {
		cbActive.RecordSuccess(ttft.Milliseconds())
	}

	// 2. Stream normalized SSE delta frames to client
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.Header().Set("X-AetherGate-Provider", activeProvider.Name())
	w.Header().Set("X-AetherGate-Tier", fmt.Sprintf("%d", modelTier))
	w.WriteHeader(http.StatusOK)

	var fullResponseBuilder strings.Builder

	// Forward first chunk if buffered
	if firstChunk != nil {
		for _, choice := range firstChunk.Choices {
			fullResponseBuilder.WriteString(choice.Delta.Content)
		}
		w.Write(models.FormatSSEChunk(firstChunk))
		if hasFlusher {
			flusher.Flush()
		}
	}

	// Forward remaining stream
	for chunk := range streamChan {
		for _, choice := range chunk.Choices {
			fullResponseBuilder.WriteString(choice.Delta.Content)
		}
		w.Write(models.FormatSSEChunk(&chunk))
		if hasFlusher {
			flusher.Flush()
		}
	}

	w.Write(models.DoneSSEChunk())
	if hasFlusher {
		flusher.Flush()
	}

	completedText := fullResponseBuilder.String()
	p.metrics.RequestsTotal.WithLabelValues(tenantID, fmt.Sprintf("tier%d", modelTier), "200").Inc()
	p.metrics.RequestDuration.WithLabelValues(fmt.Sprintf("tier%d", modelTier), "false").Observe(time.Since(overallStart).Seconds())

	// 3. Async Cache Write-Back (Background worker writes back to Redis)
	if completedText != "" && !req.SkipCache {
		go func() {
			bgCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			_ = p.cache.Store(bgCtx, tenantID, systemHash, modelTier, vec, completedText)
		}()
	}
}

// handleNonStreamingProxy handles standard non-streaming completions with failover.
func (p *ResilientProxy) handleNonStreamingProxy(
	w http.ResponseWriter,
	r *http.Request,
	req *models.ChatCompletionRequest,
	tenantID, systemHash string,
	modelTier int,
	vec []float32,
	primary, secondary Provider,
	overallStart time.Time,
) {
	primaryCtx, cancelPrimary := context.WithTimeout(r.Context(), FirstByteDeadline*4)
	defer cancelPrimary()

	resp, err := primary.Complete(primaryCtx, req)
	activeProvider := primary

	if err != nil {
		cb := p.GetCircuitBreaker(primary.Name())
		if cb != nil {
			cb.RecordFailure()
		}
		p.metrics.FailoverEventsTotal.WithLabelValues(primary.Name(), secondaryName(secondary), "error").Inc()

		if secondary != nil {
			activeProvider = secondary
			resp, err = secondary.Complete(r.Context(), req)
		}
	}

	if err != nil {
		http.Error(w, "All providers failed: "+err.Error(), http.StatusBadGateway)
		return
	}

	cbActive := p.GetCircuitBreaker(activeProvider.Name())
	if cbActive != nil {
		cbActive.RecordSuccess(time.Since(overallStart).Milliseconds())
	}

	resp.RoutingTier = modelTier
	resp.SelectedProvider = activeProvider.Name()

	p.metrics.RequestsTotal.WithLabelValues(tenantID, fmt.Sprintf("tier%d", modelTier), "200").Inc()
	p.metrics.RequestDuration.WithLabelValues(fmt.Sprintf("tier%d", modelTier), "false").Observe(time.Since(overallStart).Seconds())

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-AetherGate-Provider", activeProvider.Name())
	w.Header().Set("X-AetherGate-Tier", fmt.Sprintf("%d", modelTier))
	json.NewEncoder(w).Encode(resp)

	// Async Cache Write-Back
	if len(resp.Choices) > 0 && resp.Choices[0].Message.Content != "" && !req.SkipCache {
		content := resp.Choices[0].Message.Content
		go func() {
			bgCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
			defer cancel()
			_ = p.cache.Store(bgCtx, tenantID, systemHash, modelTier, vec, content)
		}()
	}
}

func (p *ResilientProxy) extractTenant(r *http.Request) string {
	auth := r.Header.Get("Authorization")
	var token string
	if strings.HasPrefix(auth, "Bearer ") {
		token = strings.TrimPrefix(auth, "Bearer ")
	}

	// If explicit tenant header is passed, sanitize to prevent injection or directory pathing
	if t := r.Header.Get("X-Tenant-ID"); t != "" {
		sanitized := sanitizeIdentifier(t)
		if sanitized != "" {
			return sanitized
		}
	}

	if token != "" {
		parts := strings.Split(token, "-")
		if len(parts) > 1 {
			sanitized := sanitizeIdentifier(parts[0])
			if sanitized != "" {
				return sanitized
			}
		}
		sanitized := sanitizeIdentifier(token)
		if sanitized != "" {
			return sanitized
		}
	}
	return "default-tenant"
}

// sanitizeIdentifier restricts strings to safe alphanumeric and delimiter characters [a-zA-Z0-9_.-]
// with a maximum length of 64 characters to eliminate command/query injection attacks.
func sanitizeIdentifier(s string) string {
	var sb strings.Builder
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '_' || r == '-' || r == '.' {
			sb.WriteRune(r)
		}
		if sb.Len() >= 64 {
			break
		}
	}
	return sb.String()
}

func secondaryName(s Provider) string {
	if s == nil {
		return "none"
	}
	return s.Name()
}
