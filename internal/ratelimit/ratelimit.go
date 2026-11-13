package ratelimit

import (
	"context"
	"errors"
	"math"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
)

// ErrRateLimitExceeded is returned when either RPM or TPM limits are breached.
var ErrRateLimitExceeded = errors.New("rate limit exceeded: RPM or TPM threshold breached")

// DualRateLimiter provides atomic token-bucket rate limiting for requests and tokens.
type DualRateLimiter interface {
	Allow(ctx context.Context, tenantID string, estimatedTokens int, rpm, tpm float64) (bool, error)
	Reset(ctx context.Context, tenantID string) error
}

// RedisDualRateLimiter executes atomic Redis Lua scripts via EVALSHA for sub-millisecond rate checks.
type RedisDualRateLimiter struct {
	rdb        *redis.Client
	scriptSHA  string
	scriptCode string
	mu         sync.RWMutex
}

// TokenBucketLuaScript is the exact dual-dimensional token-bucket script executed via EVALSHA.
const TokenBucketLuaScript = `
local req_key = KEYS[1]
local tok_key = KEYS[2]
local now = tonumber(ARGV[1])
local est_tokens = tonumber(ARGV[2])
local rpm_rate = tonumber(ARGV[3])
local tpm_rate = tonumber(ARGV[4])

local req_tokens = tonumber(redis.call("HGET", req_key, "tokens")) or rpm_rate
local tok_tokens = tonumber(redis.call("HGET", tok_key, "tokens")) or tpm_rate
local last_updated = tonumber(redis.call("HGET", req_key, "ts")) or now

local elapsed = math.max(0, (now - last_updated) / 60000.0)
req_tokens = math.min(rpm_rate, req_tokens + (elapsed * rpm_rate))
tok_tokens = math.min(tpm_rate, tok_tokens + (elapsed * tpm_rate))

if req_tokens < 1 or tok_tokens < est_tokens then
    return 0
end

redis.call("HSET", req_key, "tokens", req_tokens - 1, "ts", now)
redis.call("HSET", tok_key, "tokens", tok_tokens - est_tokens, "ts", now)
redis.call("PEXPIRE", req_key, 60000)
redis.call("PEXPIRE", tok_key, 60000)
return 1
`

// NewRedisDualRateLimiter loads and compiles the Lua script on the Redis server.
func NewRedisDualRateLimiter(rdb *redis.Client) (*RedisDualRateLimiter, error) {
	limiter := &RedisDualRateLimiter{
		rdb:        rdb,
		scriptCode: TokenBucketLuaScript,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	sha, err := rdb.ScriptLoad(ctx, TokenBucketLuaScript).Result()
	if err != nil {
		return nil, err
	}
	limiter.scriptSHA = sha
	return limiter, nil
}

// Allow checks if the given tenant can consume 1 request and estimatedTokens under their RPM & TPM limits.
func (r *RedisDualRateLimiter) Allow(ctx context.Context, tenantID string, estimatedTokens int, rpm, tpm float64) (bool, error) {
	if rpm <= 0 || tpm <= 0 {
		return true, nil
	}

	nowMs := time.Now().UnixNano() / int64(time.Millisecond)
	reqKey := "ratelimit:" + tenantID + ":rpm"
	tokKey := "ratelimit:" + tenantID + ":tpm"

	r.mu.RLock()
	sha := r.scriptSHA
	r.mu.RUnlock()

	res, err := r.rdb.EvalSha(ctx, sha, []string{reqKey, tokKey}, nowMs, estimatedTokens, rpm, tpm).Result()
	if err != nil {
		// If NOSCRIPT error, re-load and execute via Eval
		if redis.HasErrorPrefix(err, "NOSCRIPT") {
			r.mu.Lock()
			newSHA, loadErr := r.rdb.ScriptLoad(ctx, r.scriptCode).Result()
			if loadErr == nil {
				r.scriptSHA = newSHA
			}
			r.mu.Unlock()

			res, err = r.rdb.Eval(ctx, r.scriptCode, []string{reqKey, tokKey}, nowMs, estimatedTokens, rpm, tpm).Result()
			if err != nil {
				return false, err
			}
		} else {
			return false, err
		}
	}

	allowed, ok := res.(int64)
	if !ok {
		return false, errors.New("unexpected return type from rate limit lua script")
	}

	return allowed == 1, nil
}

// Reset clears the rate limit keys for the specified tenant.
func (r *RedisDualRateLimiter) Reset(ctx context.Context, tenantID string) error {
	reqKey := "ratelimit:" + tenantID + ":rpm"
	tokKey := "ratelimit:" + tenantID + ":tpm"
	return r.rdb.Del(ctx, reqKey, tokKey).Err()
}

// MemoryDualRateLimiter is a lock-based, zero-dependency in-memory implementation of the dual-bucket rate limiter.
type MemoryDualRateLimiter struct {
	mu      sync.Mutex
	buckets map[string]*tenantBucket
}

type tenantBucket struct {
	reqTokens float64
	tokTokens float64
	lastTime  time.Time
}

// NewMemoryDualRateLimiter creates an in-memory dual rate limiter.
func NewMemoryDualRateLimiter() *MemoryDualRateLimiter {
	return &MemoryDualRateLimiter{
		buckets: make(map[string]*tenantBucket),
	}
}

// Allow evaluates and consumes tokens from the in-memory bucket.
func (m *MemoryDualRateLimiter) Allow(ctx context.Context, tenantID string, estimatedTokens int, rpm, tpm float64) (bool, error) {
	if rpm <= 0 || tpm <= 0 {
		return true, nil
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	now := time.Now()
	b, exists := m.buckets[tenantID]
	if !exists {
		b = &tenantBucket{
			reqTokens: rpm,
			tokTokens: tpm,
			lastTime:  now,
		}
		m.buckets[tenantID] = b
	}

	elapsedMinutes := now.Sub(b.lastTime).Minutes()
	if elapsedMinutes > 0 {
		b.reqTokens = math.Min(rpm, b.reqTokens+(elapsedMinutes*rpm))
		b.tokTokens = math.Min(tpm, b.tokTokens+(elapsedMinutes*tpm))
		b.lastTime = now
	}

	if b.reqTokens < 1 || b.tokTokens < float64(estimatedTokens) {
		return false, nil
	}

	b.reqTokens -= 1
	b.tokTokens -= float64(estimatedTokens)
	return true, nil
}

// Reset clears the in-memory bucket for a tenant.
func (m *MemoryDualRateLimiter) Reset(ctx context.Context, tenantID string) error {
	m.mu.Lock()
	delete(m.buckets, tenantID)
	m.mu.Unlock()
	return nil
}
