package cache

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/aethergate/aethergate/internal/embedding"
	"github.com/aethergate/aethergate/internal/models"
	"github.com/google/uuid"
	"github.com/redis/go-redis/v9"
)

const (
	DefaultIndexName   = "idx:aethergate:cache"
	DefaultThreshold   = 0.08 // Cosine distance <= 0.08 corresponds to >= 0.92 similarity
	DefaultTTL         = 3600 * time.Second
	DefaultVectorDim   = 384
	CacheKeyPrefix     = "cache:aethergate:"
)

var (
	ErrCacheMiss     = errors.New("cache miss: no candidate below similarity threshold")
	ErrIndexCreation = errors.New("failed to initialize redis vector index")
)

// CacheEntry holds the cached response payload and metadata.
type CacheEntry struct {
	Key         string    `json:"key"`
	Response    string    `json:"response"`
	TenantID    string    `json:"tenant_id"`
	ModelTier   int       `json:"model_tier"`
	SystemHash  string    `json:"system_hash"`
	Distance    float64   `json:"distance"`
	CreatedAt   time.Time `json:"created_at"`
}

// SemanticCacheEngine defines the contract for semantic caching and replay.
type SemanticCacheEngine interface {
	Lookup(ctx context.Context, tenantID, systemHash string, modelTier int, vec []float32) (*CacheEntry, bool, error)
	Store(ctx context.Context, tenantID, systemHash string, modelTier int, vec []float32, response string) error
	ReplayStreaming(ctx context.Context, entry *CacheEntry, model string, outChan chan<- models.StreamChunk, errChan chan<- error)
	Close() error
}

// Float32ToBytes converts a float32 slice into little-endian bytes for Redis RediSearch KNN.
func Float32ToBytes(vec []float32) []byte {
	buf := make([]byte, 4*len(vec))
	for i, v := range vec {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(v))
	}
	return buf
}

// BytesToFloat32 converts little-endian bytes back into a float32 slice.
func BytesToFloat32(buf []byte) []float32 {
	if len(buf)%4 != 0 {
		return nil
	}
	vec := make([]float32, len(buf)/4)
	for i := range vec {
		bits := binary.LittleEndian.Uint32(buf[i*4 : (i+1)*4])
		vec[i] = math.Float32frombits(bits)
	}
	return vec
}

// EscapeRediSearchTag escapes all special characters in RediSearch TAG queries
// to prevent query injection and cross-tenant data leakage.
// Special characters in RediSearch: , . < > { } [ ] " ' : ; ! @ # $ % ^ & * ( ) - + = ~ | / \
func EscapeRediSearchTag(s string) string {
	var sb strings.Builder
	for _, r := range s {
		switch r {
		case ',', '.', '<', '>', '{', '}', '[', ']', '"', '\'', ':', ';',
			'!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '+', '=',
			'~', '|', '/', '\\', ' ':
			sb.WriteRune('\\')
			sb.WriteRune(r)
		default:
			sb.WriteRune(r)
		}
	}
	return sb.String()
}

// HashSystemPrompt produces a compact SHA256 hex string of the system prompt to prevent cross-persona pollution.
func HashSystemPrompt(systemPrompt string) string {
	if systemPrompt == "" {
		return "none"
	}
	h := sha256.Sum256([]byte(systemPrompt))
	return hex.EncodeToString(h[:8])
}

// RedisSemanticCache implements sub-10ms Redis Stack HNSW vector search.
type RedisSemanticCache struct {
	rdb       *redis.Client
	indexName string
	threshold float64
	ttl       time.Duration
}

// NewRedisSemanticCache constructs a Redis Stack backed semantic cache and ensures the HNSW index exists.
func NewRedisSemanticCache(rdb *redis.Client, indexName string, threshold float64, ttl time.Duration) (*RedisSemanticCache, error) {
	if indexName == "" {
		indexName = DefaultIndexName
	}
	if threshold <= 0 {
		threshold = DefaultThreshold
	}
	if ttl <= 0 {
		ttl = DefaultTTL
	}

	sc := &RedisSemanticCache{
		rdb:       rdb,
		indexName: indexName,
		threshold: threshold,
		ttl:       ttl,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_ = sc.ensureIndex(ctx)
	return sc, nil
}

// ensureIndex verifies or creates the RediSearch HNSW vector index with multi-tenant tag filters.
func (s *RedisSemanticCache) ensureIndex(ctx context.Context) error {
	// Check if index already exists
	info := s.rdb.Do(ctx, "FT.INFO", s.indexName)
	if info.Err() == nil {
		return nil
	}

	// FT.CREATE idx:aethergate:cache ON HASH PREFIX 1 cache:aethergate: SCHEMA
	// tenant_id TAG system_hash TAG model_tier NUMERIC
	// embedding VECTOR HNSW 6 TYPE FLOAT32 DIM 384 DISTANCE_METRIC COSINE
	args := []interface{}{
		"FT.CREATE", s.indexName,
		"ON", "HASH",
		"PREFIX", "1", CacheKeyPrefix,
		"SCHEMA",
		"tenant_id", "TAG",
		"system_hash", "TAG",
		"model_tier", "NUMERIC",
		"response", "TEXT",
		"embedding", "VECTOR", "HNSW", "6",
		"TYPE", "FLOAT32",
		"DIM", strconv.Itoa(DefaultVectorDim),
		"DISTANCE_METRIC", "COSINE",
	}

	err := s.rdb.Do(ctx, args...).Err()
	if err != nil && !strings.Contains(err.Error(), "Index already exists") {
		return fmt.Errorf("%w: %v", ErrIndexCreation, err)
	}
	return nil
}

// Lookup executes an HNSW KNN query scoped by tenant_id, system_hash, and optionally model_tier.
func (s *RedisSemanticCache) Lookup(ctx context.Context, tenantID, systemHash string, modelTier int, vec []float32) (*CacheEntry, bool, error) {
	if systemHash == "" {
		systemHash = "none"
	}

	// Sanitize and escape tag parameters to prevent RediSearch query injection
	safeTenant := EscapeRediSearchTag(tenantID)
	safeHash := EscapeRediSearchTag(systemHash)

	// Scoped KNN query using dialect 2
	var filter string
	if modelTier > 0 {
		filter = fmt.Sprintf("(@tenant_id:{%s} @system_hash:{%s} @model_tier:[%d %d])", safeTenant, safeHash, modelTier, modelTier)
	} else {
		filter = fmt.Sprintf("(@tenant_id:{%s} @system_hash:{%s})", safeTenant, safeHash)
	}
	query := fmt.Sprintf("%s=>[KNN 1 @embedding $blob AS dist]", filter)

	res, err := s.rdb.Do(ctx,
		"FT.SEARCH", s.indexName, query,
		"PARAMS", "2", "blob", Float32ToBytes(vec),
		"SORTBY", "dist",
		"RETURN", "3", "response", "dist", "tenant_id",
		"DIALECT", "2",
	).Result()
	if err != nil {
		return nil, false, err
	}

	resSlice, ok := res.([]interface{})
	if !ok || len(resSlice) < 2 {
		return nil, false, nil
	}

	count, ok := resSlice[0].(int64)
	if !ok || count == 0 {
		return nil, false, nil
	}

	docKey, _ := resSlice[1].(string)
	docFields, ok := resSlice[2].([]interface{})
	if !ok {
		return nil, false, nil
	}

	var responseText string
	var distVal float64 = 1.0

	for i := 0; i < len(docFields); i += 2 {
		k, _ := docFields[i].(string)
		if i+1 < len(docFields) {
			v := docFields[i+1]
			switch k {
			case "response":
				responseText, _ = v.(string)
			case "dist":
				if s, ok := v.(string); ok {
					distVal, _ = strconv.ParseFloat(s, 64)
				}
			}
		}
	}

	// Verify candidate cosine distance is below threshold (0.08)
	if distVal > s.threshold {
		return nil, false, nil
	}

	entry := &CacheEntry{
		Key:        docKey,
		Response:   responseText,
		TenantID:   tenantID,
		ModelTier:  modelTier,
		SystemHash: systemHash,
		Distance:   distVal,
		CreatedAt:  time.Now(),
	}

	return entry, true, nil
}

// Store writes the prompt embedding and generated response into Redis with a 3,600s TTL.
func (s *RedisSemanticCache) Store(ctx context.Context, tenantID, systemHash string, modelTier int, vec []float32, response string) error {
	if systemHash == "" {
		systemHash = "none"
	}
	key := fmt.Sprintf("%s%s:%s", CacheKeyPrefix, tenantID, uuid.New().String())

	pipe := s.rdb.Pipeline()
	pipe.HSet(ctx, key, map[string]interface{}{
		"tenant_id":   tenantID,
		"system_hash": systemHash,
		"model_tier":  modelTier,
		"response":    response,
		"embedding":   Float32ToBytes(vec),
		"created_at":  time.Now().Unix(),
	})
	pipe.Expire(ctx, key, s.ttl)

	_, err := pipe.Exec(ctx)
	return err
}

// ReplayStreaming micro-batches the cached response and simulates low-latency SSE token streaming.
func (s *RedisSemanticCache) ReplayStreaming(ctx context.Context, entry *CacheEntry, model string, outChan chan<- models.StreamChunk, errChan chan<- error) {
	ReplayStreamTokens(ctx, entry.Response, model, outChan, errChan)
}

func (s *RedisSemanticCache) Close() error {
	return nil
}

// MemorySemanticCache is a thread-safe in-memory vector cache used for standalone runs and unit testing.
type MemorySemanticCache struct {
	mu        sync.RWMutex
	entries   map[string]*memCacheItem
	threshold float64
	ttl       time.Duration
}

type memCacheItem struct {
	entry     *CacheEntry
	embedding []float32
	expiresAt time.Time
}

// NewMemorySemanticCache constructs an in-memory vector cache.
func NewMemorySemanticCache(threshold float64, ttl time.Duration) *MemorySemanticCache {
	if threshold <= 0 {
		threshold = DefaultThreshold
	}
	if ttl <= 0 {
		ttl = DefaultTTL
	}
	return &MemorySemanticCache{
		entries:   make(map[string]*memCacheItem),
		threshold: threshold,
		ttl:       ttl,
	}
}

func (m *MemorySemanticCache) Lookup(ctx context.Context, tenantID, systemHash string, modelTier int, vec []float32) (*CacheEntry, bool, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	now := time.Now()
	var bestEntry *CacheEntry
	bestDist := math.MaxFloat64

	for _, item := range m.entries {
		if now.After(item.expiresAt) {
			continue
		}
		// Multi-tenant and persona scoping
		if item.entry.TenantID != tenantID || item.entry.SystemHash != systemHash {
			continue
		}
		if modelTier > 0 && item.entry.ModelTier != modelTier {
			continue
		}

		dist, err := embedding.CosineDistance(vec, item.embedding)
		if err != nil {
			continue
		}

		if dist < bestDist {
			bestDist = dist
			bestEntry = item.entry
		}
	}

	if bestEntry != nil && bestDist <= m.threshold {
		copyEntry := *bestEntry
		copyEntry.Distance = bestDist
		return &copyEntry, true, nil
	}

	return nil, false, nil
}

func (m *MemorySemanticCache) Store(ctx context.Context, tenantID, systemHash string, modelTier int, vec []float32, response string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	key := fmt.Sprintf("%s:%s", tenantID, uuid.New().String())
	entry := &CacheEntry{
		Key:        key,
		Response:   response,
		TenantID:   tenantID,
		ModelTier:  modelTier,
		SystemHash: systemHash,
		CreatedAt:  time.Now(),
	}

	vecCopy := make([]float32, len(vec))
	copy(vecCopy, vec)

	m.entries[key] = &memCacheItem{
		entry:     entry,
		embedding: vecCopy,
		expiresAt: time.Now().Add(m.ttl),
	}
	return nil
}

func (m *MemorySemanticCache) ReplayStreaming(ctx context.Context, entry *CacheEntry, model string, outChan chan<- models.StreamChunk, errChan chan<- error) {
	ReplayStreamTokens(ctx, entry.Response, model, outChan, errChan)
}

func (m *MemorySemanticCache) Close() error {
	return nil
}

// ReplayStreamTokens splits a completed response into micro-batches and streams them as OpenAI SSE chunks.
func ReplayStreamTokens(ctx context.Context, text, model string, outChan chan<- models.StreamChunk, errChan chan<- error) {
	defer close(outChan)

	// Break text into token-sized words/chunks
	words := strings.Split(text, " ")
	chunkID := "chatcmpl-cached-" + uuid.New().String()
	created := time.Now().Unix()

	// Initial role chunk
	initialChunk := models.StreamChunk{
		ID:      chunkID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []models.StreamChoice{
			{
				Index: 0,
				Delta: models.Delta{
					Role: models.RoleAssistant,
				},
			},
		},
	}

	select {
	case <-ctx.Done():
		if errChan != nil {
			errChan <- ctx.Err()
		}
		return
	case outChan <- initialChunk:
	}

	// Micro-batch tokens (2-3 words per chunk)
	batchSize := 2
	for i := 0; i < len(words); i += batchSize {
		select {
		case <-ctx.Done():
			if errChan != nil {
				errChan <- ctx.Err()
			}
			return
		default:
		}

		end := i + batchSize
		if end > len(words) {
			end = len(words)
		}

		deltaText := strings.Join(words[i:end], " ")
		if end < len(words) {
			deltaText += " "
		}

		chunk := models.StreamChunk{
			ID:      chunkID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   model,
			Choices: []models.StreamChoice{
				{
					Index: 0,
					Delta: models.Delta{
						Content: deltaText,
					},
				},
			},
		}

		outChan <- chunk
		// Simulate realistic sub-10ms micro-batch streaming cadence
		time.Sleep(2 * time.Millisecond)
	}

	// Final terminating chunk with finish_reason: "stop"
	finalChunk := models.StreamChunk{
		ID:      chunkID,
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   model,
		Choices: []models.StreamChoice{
			{
				Index:        0,
				Delta:        models.Delta{},
				FinishReason: "stop",
			},
		},
	}
	outChan <- finalChunk
}

// CacheSerializer converts CacheEntry to/from JSON for telemetry and diagnostics.
func (e *CacheEntry) ToJSON() []byte {
	b, _ := json.Marshal(e)
	return b
}
