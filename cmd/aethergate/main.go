package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/aethergate/aethergate/internal/cache"
	"github.com/aethergate/aethergate/internal/embedding"
	"github.com/aethergate/aethergate/internal/proxy"
	"github.com/aethergate/aethergate/internal/ratelimit"
	"github.com/aethergate/aethergate/internal/router"
	"github.com/aethergate/aethergate/internal/telemetry"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/redis/go-redis/v9"
)

type Config struct {
	Port           int
	RedisAddr      string
	RedisPassword  string
	GeminiAPIKey   string
	VLLMURL        string
	VLLMAPIKey     string
	BedrockRegion  string
	BedrockAccess  string
	BedrockSecret  string
	DefaultRPM     float64
	DefaultTPM     float64
	FastThreshold  float64
	FlashThreshold float64
	WorkerPoolSize int
}

func loadConfig() Config {
	cfg := Config{
		Port:           getEnvInt("PORT", 8080),
		RedisAddr:      getEnv("REDIS_URL", "localhost:6379"),
		RedisPassword:  getEnv("REDIS_PASSWORD", ""),
		GeminiAPIKey:   getEnv("GEMINI_API_KEY", ""),
		VLLMURL:        getEnv("VLLM_URL", "http://localhost:8000"),
		VLLMAPIKey:     getEnv("VLLM_API_KEY", ""),
		BedrockRegion:  getEnv("AWS_REGION", "us-east-1"),
		BedrockAccess:  getEnv("AWS_ACCESS_KEY_ID", ""),
		BedrockSecret:  getEnv("AWS_SECRET_ACCESS_KEY", ""),
		DefaultRPM:     getEnvFloat("DEFAULT_RPM", 600.0),
		DefaultTPM:     getEnvFloat("DEFAULT_TPM", 500000.0),
		FastThreshold:  getEnvFloat("FAST_THRESHOLD", 0.35),
		FlashThreshold: getEnvFloat("FLASH_THRESHOLD", 0.70),
		WorkerPoolSize: getEnvInt("WORKER_POOL_SIZE", 8),
	}

	flag.IntVar(&cfg.Port, "port", cfg.Port, "HTTP server listening port")
	flag.StringVar(&cfg.RedisAddr, "redis", cfg.RedisAddr, "Redis Stack connection address")
	flag.Float64Var(&cfg.FastThreshold, "fast-threshold", cfg.FastThreshold, "RouteLLM fast tier cutoff threshold")
	flag.Float64Var(&cfg.FlashThreshold, "flash-threshold", cfg.FlashThreshold, "RouteLLM flash tier cutoff threshold")
	flag.Parse()

	return cfg
}

func main() {
	cfg := loadConfig()

	log.Println("================================================================================")
	log.Println(" AetherGate - OpenAI-Compatible AI Gateway & Semantic Load Balancer")
	log.Println(" Sub-10ms Redis HNSW Cache | Local ONNX Routing | Resilient SSE Failover")
	log.Println("================================================================================")

	// 1. Telemetry and Prometheus Metrics
	promRegistry := prometheus.NewRegistry()
	metrics := telemetry.NewMetrics(promRegistry)
	promRegistry.MustRegister(prometheus.NewGoCollector())
	promRegistry.MustRegister(prometheus.NewProcessCollector(prometheus.ProcessCollectorOpts{}))

	// 2. In-Process ONNX Embedding Engine & Worker Pool
	log.Printf("[Init] Starting In-Process 384-d Embedding Engine (Workers: %d)...", cfg.WorkerPoolSize)
	embedder := embedding.NewFastMiniLMEmbedder()
	workerPool := embedding.NewWorkerPool(embedder, cfg.WorkerPoolSize, 512)
	defer workerPool.Close()

	// 3. Redis Stack Connection & Semantic Cache / Rate Limiter
	var cacheEngine cache.SemanticCacheEngine
	var rateLimiter ratelimit.DualRateLimiter

	rdb := redis.NewClient(&redis.Options{
		Addr:     cfg.RedisAddr,
		Password: cfg.RedisPassword,
	})

	pingCtx, pingCancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer pingCancel()

	if err := rdb.Ping(pingCtx).Err(); err != nil {
		log.Printf("[Warn] Redis connection to %s failed (%v). Falling back to In-Memory HNSW vector cache & rate limiter.", cfg.RedisAddr, err)
		cacheEngine = cache.NewMemorySemanticCache(0.08, 3600*time.Second)
		rateLimiter = ratelimit.NewMemoryDualRateLimiter()
	} else {
		log.Printf("[Init] Connected to Redis Stack at %s. Initializing HNSW vector index...", cfg.RedisAddr)
		rc, err := cache.NewRedisSemanticCache(rdb, cache.DefaultIndexName, 0.08, 3600*time.Second)
		if err != nil {
			log.Printf("[Warn] Redis HNSW index init failed (%v). Falling back to Memory Cache.", err)
			cacheEngine = cache.NewMemorySemanticCache(0.08, 3600*time.Second)
		} else {
			cacheEngine = rc
			log.Println("[Init] Redis Stack HNSW vector index verified.")
		}

		rl, err := ratelimit.NewRedisDualRateLimiter(rdb)
		if err != nil {
			log.Printf("[Warn] Redis Lua rate limiter script load failed (%v). Falling back to Memory Limiter.", err)
			rateLimiter = ratelimit.NewMemoryDualRateLimiter()
		} else {
			rateLimiter = rl
			log.Println("[Init] Redis Lua Dual-Bucket Rate Limiter loaded.")
		}
	}

	// 4. Berkeley RouteLLM Complexity Router
	routerCfg := router.RouterConfig{
		FastThreshold:  cfg.FastThreshold,
		FlashThreshold: cfg.FlashThreshold,
	}
	complexityRouter := router.NewComplexityRouter(routerCfg)
	log.Printf("[Init] RouteLLM matrix-factorization router initialized (Fast < %.2f, Flash < %.2f, Frontier >= %.2f)",
		cfg.FastThreshold, cfg.FlashThreshold, cfg.FlashThreshold)

	// 5. Reverse Proxy & Upstream Providers
	gatewayCfg := proxy.GatewayConfig{
		DefaultRPM: cfg.DefaultRPM,
		DefaultTPM: cfg.DefaultTPM,
	}
	gatewayProxy := proxy.NewResilientProxy(gatewayCfg, workerPool, cacheEngine, complexityRouter, rateLimiter, metrics)

	// Register Provider Adapters
	// Tier 1: Local Fast (vLLM / Llama-3-8B)
	vllmProvider := proxy.NewVLLMProvider("vllm-llama3-8b", router.Tier1LocalFast, cfg.VLLMURL, cfg.VLLMAPIKey)
	gatewayProxy.RegisterProvider(vllmProvider)

	// Backup mock fast provider for resilience
	mockFast := proxy.NewMockProvider("mock-fast-backup", router.Tier1LocalFast).
		SetResponseBody("Response from Tier 1 Local Fast fallback.")
	gatewayProxy.RegisterProvider(mockFast)

	// Tier 2: Cloud Flash (Gemini 2.5 Flash / Haiku)
	geminiProvider := proxy.NewGeminiProvider("gemini-2.5-flash", router.Tier2CloudFlash, "gemini-2.5-flash", cfg.GeminiAPIKey)
	gatewayProxy.RegisterProvider(geminiProvider)

	mockFlash := proxy.NewMockProvider("mock-flash-backup", router.Tier2CloudFlash).
		SetResponseBody("Response from Tier 2 Cloud Flash fallback.")
	gatewayProxy.RegisterProvider(mockFlash)

	// Tier 3: Frontier (AWS Bedrock / Claude Sonnet)
	bedrockProvider := proxy.NewBedrockProvider("bedrock-claude-sonnet", router.Tier3Frontier, "anthropic.claude-3-5-sonnet-20241022-v2:0", cfg.BedrockRegion, cfg.BedrockAccess, cfg.BedrockSecret)
	gatewayProxy.RegisterProvider(bedrockProvider)

	mockFrontier := proxy.NewMockProvider("mock-frontier-backup", router.Tier3Frontier).
		SetResponseBody("Response from Tier 3 Frontier Claude Sonnet fallback.")
	gatewayProxy.RegisterProvider(mockFrontier)

	log.Println("[Init] Registered providers across Tier 1 (vLLM), Tier 2 (Gemini), Tier 3 (Bedrock).")

	// 6. HTTP Routing Mux
	mux := http.NewServeMux()

	// OpenAI-compatible Chat Completions endpoint
	mux.Handle("/v1/chat/completions", gatewayProxy)

	// Models listing endpoint
	mux.HandleFunc("/v1/models", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"object": "list",
			"data": []map[string]interface{}{
				{"id": "aethergate-auto", "object": "model", "owned_by": "aethergate", "routing": "dynamic_complexity"},
				{"id": "tier-1-fast", "object": "model", "owned_by": "vllm-llama3-8b", "cost_per_1m": 0.05},
				{"id": "tier-2-flash", "object": "model", "owned_by": "gemini-2.5-flash", "cost_per_1m": 0.15},
				{"id": "tier-3-frontier", "object": "model", "owned_by": "bedrock-claude-sonnet", "cost_per_1m": 3.00},
			},
		})
	})

	// Health check endpoint
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":    "healthy",
			"timestamp": time.Now().UTC().Format(time.RFC3339),
			"version":   "1.0.0",
		})
	})

	// Prometheus Metrics endpoint
	mux.Handle("/metrics", promhttp.HandlerFor(promRegistry, promhttp.HandlerOpts{}))

	// Circuit breaker stats inspection endpoint
	mux.HandleFunc("/v1/circuitbreakers", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		stats := make(map[string]interface{})
		for _, name := range []string{"vllm-llama3-8b", "gemini-2.5-flash", "bedrock-claude-sonnet", "mock-fast-backup", "mock-flash-backup", "mock-frontier-backup"} {
			if cb := gatewayProxy.GetCircuitBreaker(name); cb != nil {
				stats[name] = cb.GetStats()
			}
		}
		json.NewEncoder(w).Encode(stats)
	})

	server := &http.Server{
		Addr:              fmt.Sprintf(":%d", cfg.Port),
		Handler:           securityHeadersMiddleware(mux),
		ReadHeaderTimeout: 5 * time.Second, // Mitigates Slowloris header exhaustion attacks
		ReadTimeout:       15 * time.Second,
		WriteTimeout:      120 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	// 7. Start Server & Graceful Shutdown
	go func() {
		log.Printf("[Server] AetherGate HTTP Server listening on port :%d", cfg.Port)
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("[Fatal] Server terminated unexpectedly: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("[Server] Gracefully shutting down AetherGate...")
	shutdownCtx, cancelShutdown := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancelShutdown()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Printf("[Error] Server shutdown failed: %v", err)
	}
	log.Println("[Server] AetherGate successfully stopped.")
}

func getEnv(key, defaultVal string) string {
	if val, ok := os.LookupEnv(key); ok && val != "" {
		return val
	}
	return defaultVal
}

func getEnvInt(key string, defaultVal int) int {
	if val, ok := os.LookupEnv(key); ok {
		if i, err := strconv.Atoi(val); err == nil {
			return i
		}
	}
	return defaultVal
}

func getEnvFloat(key string, defaultVal float64) float64 {
	if val, ok := os.LookupEnv(key); ok {
		if f, err := strconv.ParseFloat(val, 64); err == nil {
			return f
		}
	}
	return defaultVal
}

// securityHeadersMiddleware enforces baseline OWASP security headers across all HTTP responses.
func securityHeadersMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("Referrer-Policy", "strict-origin-when-cross-origin")
		w.Header().Set("X-XSS-Protection", "1; mode=block")
		next.ServeHTTP(w, r)
	})
}
