import http from 'k6/http';
import { check, sleep } from 'k6';
import { Counter, Rate, Trend } from 'k6/metrics';

// Custom Prometheus-compatible metrics
export const cacheHitRate = new Rate('aethergate_cache_hit_rate');
export const failoverCount = new Counter('aethergate_failover_count');
export const ttftTrend = new Trend('aethergate_ttft_ms');

export const options = {
  scenarios: {
    rag_workload: {
      executor: 'ramping-vus',
      startVUs: 5,
      stages: [
        { duration: '30s', target: 50 },  // Ramp-up to 50 concurrent agents
        { duration: '1m', target: 50 },   // Sustain high load
        { duration: '30s', target: 0 },   // Ramp-down
      ],
      gracefulRampDown: '10s',
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<400', 'p(99)<1200'], // P95 under 400ms, P99 under 1200ms
    http_req_failed: ['rate<0.02'],                  // Less than 2% errors under load
  },
};

const syntheticPrompts = [
  "How do I set up mutual TLS authentication with client certificates in Kubernetes?",
  "Summarize the quarterly SOC2 compliance findings for AWS RDS Aurora clusters.",
  "Extract invoice total, vendor name, and date from the following raw text: Invoice #99182...",
  "Classify this incoming customer ticket as Billing, Technical, or Account Access.",
  "```python\ndef solve_knapsack(weights, values, capacity):\n``` Explain step by step how to prove the dynamic programming recurrence and derive the time complexity.",
  "What is the capital of France?",
  "Translate the following user review to Spanish: 'The product arrived on time and works well.'",
  "Analyze the architectural trade-offs between Redis HNSW vector caching and pgvector under high write throughput."
];

export default function () {
  // 35% probability of repeating a common query to test Semantic Cache HNSW hit rate
  let prompt;
  if (Math.random() < 0.35) {
    prompt = syntheticPrompts[0];
  } else {
    prompt = syntheticPrompts[Math.floor(Math.random() * syntheticPrompts.length)];
  }

  const tenantID = `tenant-${Math.floor(Math.random() * 5) + 1}`;
  const url = __ENV.TARGET_URL || 'http://localhost:8080/v1/chat/completions';

  const payload = JSON.stringify({
    model: 'gpt-4o',
    stream: false,
    messages: [
      { role: 'user', content: prompt }
    ]
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
      'X-Tenant-ID': tenantID,
      'Authorization': `Bearer ${tenantID}-secret-key`
    },
    timeout: '10s'
  };

  const res = http.post(url, payload, params);

  const isCacheHit = res.headers['X-Aethergate-Cache'] === 'HIT';
  cacheHitRate.add(isCacheHit);

  const isSuccess = check(res, {
    'status is 200': (r) => r.status === 200,
    'has valid provider': (r) => r.headers['X-Aethergate-Provider'] !== undefined,
  });

  if (!isSuccess && res.status !== 429) {
    failoverCount.add(1);
  }

  sleep(0.05); // 50ms think time
}
