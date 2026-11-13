package main

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
	"github.com/aethergate/aethergate/internal/proxy"
	"github.com/aethergate/aethergate/internal/ratelimit"
	"github.com/aethergate/aethergate/internal/router"
	"github.com/aethergate/aethergate/internal/telemetry"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
)

func setupTestServer() *httptest.Server {
	reg := prometheus.NewRegistry()
	metrics := telemetry.NewMetrics(reg)

	embedder := embedding.NewFastMiniLMEmbedder()
	workerPool := embedding.NewWorkerPool(embedder, 4, 32)
	cacheEngine := cache.NewMemorySemanticCache(0.08, time.Hour)
	complexityRouter := router.NewComplexityRouter(router.DefaultRouterConfig())
	rateLimiter := ratelimit.NewMemoryDualRateLimiter()

	gatewayProxy := proxy.NewResilientProxy(
		proxy.GatewayConfig{DefaultRPM: 600, DefaultTPM: 500000},
		workerPool,
		cacheEngine,
		complexityRouter,
		rateLimiter,
		metrics,
	)

	// Register test providers
	mockFast := proxy.NewMockProvider("mock-vllm-llama", router.Tier1LocalFast).
		SetResponseBody("Fast response from local Llama model.")
	gatewayProxy.RegisterProvider(mockFast)

	mockFlash := proxy.NewMockProvider("mock-gemini-flash", router.Tier2CloudFlash).
		SetResponseBody("Flash response from Google Gemini.")
	gatewayProxy.RegisterProvider(mockFlash)

	mockFrontier := proxy.NewMockProvider("mock-bedrock-claude", router.Tier3Frontier).
		SetResponseBody("Frontier response from Claude Sonnet.")
	gatewayProxy.RegisterProvider(mockFrontier)

	mux := http.NewServeMux()
	mux.Handle("/v1/chat/completions", gatewayProxy)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"status":"healthy"}`))
	})
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"object":"list","data":[]}`))
	})
	mux.Handle("/metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))

	return httptest.NewServer(mux)
}

func TestEndToEndPipeline(t *testing.T) {
	server := setupTestServer()
	defer server.Close()

	client := server.Client()

	// 1. Health check
	resp, err := client.Get(server.URL + "/healthz")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("health check failed: err=%v, code=%d", err, resp.StatusCode)
	}

	// 2. Models list
	resp, err = client.Get(server.URL + "/v1/models")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("models endpoint failed: err=%v, code=%d", err, resp.StatusCode)
	}

	// 3. Metrics endpoint
	resp, err = client.Get(server.URL + "/metrics")
	if err != nil || resp.StatusCode != http.StatusOK {
		t.Fatalf("metrics endpoint failed: err=%v, code=%d", err, resp.StatusCode)
	}

	// 4. Chat completion - First request (Cache Miss)
	prompt := "What is semantic caching in AetherGate?"
	reqBody := models.ChatCompletionRequest{
		Model:  "gpt-4o",
		Stream: false,
		Messages: []models.ChatMessage{
			{Role: models.RoleUser, Content: prompt},
		},
	}
	bodyBytes, _ := json.Marshal(reqBody)

	httpReq, _ := http.NewRequest(http.MethodPost, server.URL+"/v1/chat/completions", bytes.NewReader(bodyBytes))
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("X-Tenant-ID", "tenant-e2e")

	resp1, err := client.Do(httpReq)
	if err != nil || resp1.StatusCode != http.StatusOK {
		t.Fatalf("first completion failed: %v", err)
	}

	var chatResp models.ChatCompletionResponse
	json.NewDecoder(resp1.Body).Decode(&chatResp)
	if chatResp.Cached {
		t.Fatalf("first request should not be cached")
	}

	// Wait for async cache write-back
	time.Sleep(50 * time.Millisecond)

	// 5. Chat completion - Second identical request (Cache Hit!)
	httpReq2, _ := http.NewRequest(http.MethodPost, server.URL+"/v1/chat/completions", bytes.NewReader(bodyBytes))
	httpReq2.Header.Set("Content-Type", "application/json")
	httpReq2.Header.Set("X-Tenant-ID", "tenant-e2e")

	resp2, err := client.Do(httpReq2)
	if err != nil || resp2.StatusCode != http.StatusOK {
		t.Fatalf("second completion failed: %v", err)
	}
	if resp2.Header.Get("X-AetherGate-Cache") != "HIT" {
		t.Fatalf("expected X-AetherGate-Cache: HIT on second request")
	}

	// 6. Streaming SSE completion
	reqBody.Stream = true
	reqBody.Messages[0].Content = "Tell me about high availability load balancing in Go"
	streamBytes, _ := json.Marshal(reqBody)

	streamReq, _ := http.NewRequest(http.MethodPost, server.URL+"/v1/chat/completions", bytes.NewReader(streamBytes))
	streamReq.Header.Set("Content-Type", "application/json")
	streamReq.Header.Set("X-Tenant-ID", "tenant-stream-e2e")

	streamResp, err := client.Do(streamReq)
	if err != nil || streamResp.StatusCode != http.StatusOK {
		t.Fatalf("stream completion failed: %v", err)
	}

	scanner := bufio.NewScanner(streamResp.Body)
	var sawDoneMarker bool
	var streamContent string

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				sawDoneMarker = true
				break
			}
			var chunk models.StreamChunk
			if err := json.Unmarshal([]byte(data), &chunk); err == nil {
				for _, c := range chunk.Choices {
					streamContent += c.Delta.Content
				}
			}
		}
	}

	if !sawDoneMarker {
		t.Fatalf("expected [DONE] in SSE stream")
	}
	if streamContent == "" {
		t.Fatalf("expected non-empty stream content")
	}
	t.Logf("E2E Streamed successfully: %q", streamContent)
}
