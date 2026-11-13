package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/aethergate/aethergate/internal/cache"
	"github.com/aethergate/aethergate/internal/embedding"
	"github.com/aethergate/aethergate/internal/models"
	"github.com/aethergate/aethergate/internal/ratelimit"
	"github.com/aethergate/aethergate/internal/router"
	"github.com/aethergate/aethergate/internal/telemetry"
)

func setupTestProxy() (*ResilientProxy, *cache.MemorySemanticCache, *telemetry.Metrics) {
	embedder := embedding.NewFastMiniLMEmbedder()
	pool := embedding.NewWorkerPool(embedder, 2, 16)
	c := cache.NewMemorySemanticCache(0.08, time.Hour)
	r := router.NewComplexityRouter(router.DefaultRouterConfig())
	limiter := ratelimit.NewMemoryDualRateLimiter()
	m := telemetry.NewMetrics(nil)

	proxy := NewResilientProxy(GatewayConfig{DefaultRPM: 100, DefaultTPM: 100000}, pool, c, r, limiter, m)
	return proxy, c, m
}

func TestProxyStreamingSuccess(t *testing.T) {
	proxy, _, _ := setupTestProxy()

	// Register mock providers for Tier 1 and Tier 2
	mockP1 := NewMockProvider("mock-vllm-primary", router.Tier1LocalFast).
		SetResponseBody("Hello from AetherGate streaming engine.")
	proxy.RegisterProvider(mockP1)

	reqPayload := models.ChatCompletionRequest{
		Model:  "gpt-4o",
		Stream: true,
		Messages: []models.ChatMessage{
			{Role: models.RoleUser, Content: "Classify this short greeting"},
		},
	}
	body, _ := json.Marshal(reqPayload)

	httpReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Tenant-ID", "tenant-test-stream")

	w := httptest.NewRecorder()
	proxy.ServeHTTP(w, httpReq)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", resp.StatusCode)
	}

	contentType := resp.Header.Get("Content-Type")
	if !strings.Contains(contentType, "text/event-stream") {
		t.Fatalf("expected text/event-stream content type, got %s", contentType)
	}

	// Read SSE lines
	scanner := bufio.NewScanner(resp.Body)
	var fullText string
	var sawDone bool

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				sawDone = true
				break
			}
			var chunk models.StreamChunk
			if err := json.Unmarshal([]byte(data), &chunk); err == nil {
				for _, c := range chunk.Choices {
					fullText += c.Delta.Content
				}
			}
		}
	}

	if !sawDone {
		t.Fatalf("expected to receive [DONE] marker")
	}
	if !strings.Contains(fullText, "Hello from AetherGate") {
		t.Fatalf("expected full response to contain greetings, got %q", fullText)
	}
}

func TestProxyFirstByteDeadlineFailover(t *testing.T) {
	proxy, _, _ := setupTestProxy()

	// Primary provider exceeds 1,500ms first byte deadline (simulate 1,800ms lag)
	slowPrimary := NewMockProvider("slow-bedrock-primary", router.Tier3Frontier).
		SetFirstByteDelay(1800 * time.Millisecond).
		SetResponseBody("Slow response that should not be reached.")

	// Secondary provider answers in 10ms
	fastSecondary := NewMockProvider("fast-gemini-secondary", router.Tier3Frontier).
		SetFirstByteDelay(10 * time.Millisecond).
		SetResponseBody("Failover response from secondary provider.")

	proxy.RegisterProvider(slowPrimary)
	proxy.RegisterProvider(fastSecondary)

	// Complex prompt routed to Tier 3 Frontier
	reqPayload := models.ChatCompletionRequest{
		Model:  "gpt-4o",
		Stream: true,
		Messages: []models.ChatMessage{
			{
				Role: models.RoleUser,
				Content: "```go\nfunc DeepAlgorithm() {}\n```\nExplain step by step how to prove the formal correctness and time complexity recurrence.",
			},
		},
	}
	body, _ := json.Marshal(reqPayload)

	httpReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Tenant-ID", "tenant-failover")

	start := time.Now()
	w := httptest.NewRecorder()
	proxy.ServeHTTP(w, httpReq)
	elapsed := time.Since(start)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected failover to succeed with status 200, got %d", resp.StatusCode)
	}

	providerUsed := resp.Header.Get("X-AetherGate-Provider")
	if providerUsed != "fast-gemini-secondary" {
		t.Fatalf("expected provider to failover to fast-gemini-secondary, got %s", providerUsed)
	}

	t.Logf("First-byte failover succeeded in %v via provider %s", elapsed, providerUsed)

	// Verify slow primary breaker recorded failure
	cbSlow := proxy.GetCircuitBreaker("slow-bedrock-primary")
	if cbSlow == nil || cbSlow.GetStats().TotalFails == 0 {
		t.Fatalf("expected slow primary breaker to record failure")
	}
}

func TestProxySemanticCacheLifecycle(t *testing.T) {
	proxy, c, _ := setupTestProxy()

	mockP := NewMockProvider("mock-vllm", router.Tier1LocalFast).
		SetResponseBody("Cached answer for repeated query.")
	proxy.RegisterProvider(mockP)

	reqPayload := models.ChatCompletionRequest{
		Model:  "gpt-4o",
		Stream: false,
		Messages: []models.ChatMessage{
			{Role: models.RoleUser, Content: "What are the core pillars of AetherGate?"},
		},
	}
	body, _ := json.Marshal(reqPayload)

	// 1. First request - Cache Miss
	httpReq1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	httpReq1.Header.Set("X-Tenant-ID", "tenant-cache-test")
	w1 := httptest.NewRecorder()
	proxy.ServeHTTP(w1, httpReq1)

	resp1 := w1.Result()
	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("req 1 failed with status %d", resp1.StatusCode)
	}
	if resp1.Header.Get("X-AetherGate-Cache") == "HIT" {
		t.Fatalf("first request should be a cache MISS")
	}

	// Wait briefly for async cache write-back goroutine
	time.Sleep(50 * time.Millisecond)

	// 2. Second request - Cache Hit!
	httpReq2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	httpReq2.Header.Set("X-Tenant-ID", "tenant-cache-test")
	w2 := httptest.NewRecorder()
	proxy.ServeHTTP(w2, httpReq2)

	resp2 := w2.Result()
	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("req 2 failed with status %d", resp2.StatusCode)
	}
	if resp2.Header.Get("X-AetherGate-Cache") != "HIT" {
		t.Fatalf("second request must be a cache HIT, got %s", resp2.Header.Get("X-AetherGate-Cache"))
	}

	var chatResp models.ChatCompletionResponse
	json.NewDecoder(resp2.Body).Decode(&chatResp)
	if !chatResp.Cached {
		t.Fatalf("expected response.Cached to be true")
	}
	if chatResp.SelectedProvider != "redis_hnsw_cache" {
		t.Fatalf("expected SelectedProvider redis_hnsw_cache, got %s", chatResp.SelectedProvider)
	}

	_ = c
}

func TestProxyRateLimitBreach(t *testing.T) {
	embedder := embedding.NewFastMiniLMEmbedder()
	pool := embedding.NewWorkerPool(embedder, 2, 16)
	c := cache.NewMemorySemanticCache(0.08, time.Hour)
	r := router.NewComplexityRouter(router.DefaultRouterConfig())
	limiter := ratelimit.NewMemoryDualRateLimiter()
	m := telemetry.NewMetrics(nil)

	// Strict rate limit: 2 RPM
	proxy := NewResilientProxy(GatewayConfig{DefaultRPM: 2, DefaultTPM: 100000}, pool, c, r, limiter, m)
	mockP := NewMockProvider("mock-vllm", router.Tier1LocalFast)
	proxy.RegisterProvider(mockP)

	reqPayload := models.ChatCompletionRequest{
		Model: "gpt-4o",
		Messages: []models.ChatMessage{
			{Role: models.RoleUser, Content: "Hello world"},
		},
	}
	body, _ := json.Marshal(reqPayload)

	tenant := "tenant-throttled"

	// 1st request OK
	req1 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req1.Header.Set("X-Tenant-ID", tenant)
	w1 := httptest.NewRecorder()
	proxy.ServeHTTP(w1, req1)
	if w1.Code != http.StatusOK {
		t.Fatalf("req 1 failed: %d", w1.Code)
	}

	// 2nd request OK
	req2 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req2.Header.Set("X-Tenant-ID", tenant)
	w2 := httptest.NewRecorder()
	proxy.ServeHTTP(w2, req2)
	if w2.Code != http.StatusOK {
		t.Fatalf("req 2 failed: %d", w2.Code)
	}

	// 3rd request must be rejected with HTTP 429
	req3 := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	req3.Header.Set("X-Tenant-ID", tenant)
	w3 := httptest.NewRecorder()
	proxy.ServeHTTP(w3, req3)
	if w3.Code != http.StatusTooManyRequests {
		t.Fatalf("expected HTTP 429, got %d", w3.Code)
	}
}
