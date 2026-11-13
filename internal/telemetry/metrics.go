package telemetry

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// Metrics holds all AetherGate Prometheus metric collectors.
type Metrics struct {
	Registry                 *prometheus.Registry
	RequestsTotal            *prometheus.CounterVec
	CacheHitsTotal           *prometheus.CounterVec
	CacheMissesTotal         *prometheus.CounterVec
	TTFTHistogram            *prometheus.HistogramVec
	RequestDuration          *prometheus.HistogramVec
	TokensSavedTotal         *prometheus.CounterVec
	DollarsSavedTotal        *prometheus.CounterVec
	CircuitBreakerState      *prometheus.GaugeVec
	FailoverEventsTotal      *prometheus.CounterVec
	RateLimitRejectionsTotal *prometheus.CounterVec
	EmbeddingDuration        prometheus.Histogram
}

// NewMetrics initializes and registers Prometheus metrics for AetherGate in the provided registerer.
// If reg is nil, a dedicated fresh registry is created.
func NewMetrics(reg prometheus.Registerer) *Metrics {
	var registry *prometheus.Registry
	if reg == nil {
		registry = prometheus.NewRegistry()
		reg = registry
	} else if r, ok := reg.(*prometheus.Registry); ok {
		registry = r
	}

	factory := promauto.With(reg)

	m := &Metrics{
		Registry: registry,
		RequestsTotal: factory.NewCounterVec(
			prometheus.CounterOpts{
				Name: "aethergate_requests_total",
				Help: "Total number of chat completion requests processed by AetherGate",
			},
			[]string{"tenant_id", "tier", "status"},
		),
		CacheHitsTotal: factory.NewCounterVec(
			prometheus.CounterOpts{
				Name: "aethergate_cache_hits_total",
				Help: "Total number of semantic cache hits in Redis HNSW",
			},
			[]string{"tenant_id"},
		),
		CacheMissesTotal: factory.NewCounterVec(
			prometheus.CounterOpts{
				Name: "aethergate_cache_misses_total",
				Help: "Total number of semantic cache misses routed upstream",
			},
			[]string{"tenant_id"},
		),
		TTFTHistogram: factory.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "aethergate_ttft_seconds",
				Help:    "Time To First Token (TTFT) across providers and cache hits",
				Buckets: []float64{0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.000, 1.500, 2.000, 3.000},
			},
			[]string{"provider", "tier"},
		),
		RequestDuration: factory.NewHistogramVec(
			prometheus.HistogramOpts{
				Name:    "aethergate_request_duration_seconds",
				Help:    "Total end-to-end request latency including full token streaming",
				Buckets: []float64{0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0},
			},
			[]string{"tier", "cached"},
		),
		TokensSavedTotal: factory.NewCounterVec(
			prometheus.CounterOpts{
				Name: "aethergate_tokens_saved_total",
				Help: "Estimated tokens eliminated via semantic caching and local fast routing",
			},
			[]string{"tenant_id"},
		),
		DollarsSavedTotal: factory.NewCounterVec(
			prometheus.CounterOpts{
				Name: "aethergate_dollars_saved_total",
				Help: "Estimated dollars saved compared to naive 100% frontier routing",
			},
			[]string{"tenant_id"},
		),
		CircuitBreakerState: factory.NewGaugeVec(
			prometheus.GaugeOpts{
				Name: "aethergate_circuit_breaker_state",
				Help: "Circuit breaker status (0=Closed/Healthy, 1=Half-Open, 2=Open/Failing)",
			},
			[]string{"provider"},
		),
		FailoverEventsTotal: factory.NewCounterVec(
			prometheus.CounterOpts{
				Name: "aethergate_failover_events_total",
				Help: "Count of first-byte deadline failovers to secondary providers",
			},
			[]string{"primary_provider", "secondary_provider", "reason"},
		),
		RateLimitRejectionsTotal: factory.NewCounterVec(
			prometheus.CounterOpts{
				Name: "aethergate_rate_limit_rejections_total",
				Help: "Total requests rejected by atomic token bucket rate limiting",
			},
			[]string{"tenant_id", "dimension"},
		),
		EmbeddingDuration: factory.NewHistogram(
			prometheus.HistogramOpts{
				Name:    "aethergate_embedding_duration_seconds",
				Help:    "In-process ONNX MiniLM vector embedding computation latency",
				Buckets: []float64{0.0005, 0.001, 0.002, 0.003, 0.005, 0.010},
			},
		),
	}
	return m
}
