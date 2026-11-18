package models

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
)

// Role constants
const (
	RoleSystem    = "system"
	RoleUser      = "user"
	RoleAssistant = "assistant"
	RoleFunction  = "function"
	RoleTool      = "tool"
)

// ChatMessage represents a single message in a chat completion request or response.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
	Name    string `json:"name,omitempty"`
}

// ChatCompletionRequest is the OpenAI-compatible chat completion payload.
type ChatCompletionRequest struct {
	Model            string        `json:"model"`
	Messages         []ChatMessage `json:"messages"`
	Temperature      *float64      `json:"temperature,omitempty"`
	TopP             *float64      `json:"top_p,omitempty"`
	N                *int          `json:"n,omitempty"`
	Stream           bool          `json:"stream,omitempty"`
	Stop             []string      `json:"stop,omitempty"`
	MaxTokens        *int          `json:"max_tokens,omitempty"`
	PresencePenalty  *float64      `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64      `json:"frequency_penalty,omitempty"`
	User             string        `json:"user,omitempty"`

	// AetherGate specific internal metadata (injected via context / middleware)
	TenantID         string        `json:"tenant_id,omitempty"`
	ForceTier        *int          `json:"force_tier,omitempty"`
	SkipCache        bool          `json:"skip_cache,omitempty"`
}

// PromptText extracts and normalizes the full prompt text for embedding and routing.
func (r *ChatCompletionRequest) PromptText() string {
	var sb strings.Builder
	for _, m := range r.Messages {
		if m.Role == RoleUser || m.Role == RoleSystem {
			sb.WriteString(m.Content)
			sb.WriteString("\n")
		}
	}
	return strings.TrimSpace(sb.String())
}

// SystemPrompt extracts any system message if present.
func (r *ChatCompletionRequest) SystemPrompt() string {
	for _, m := range r.Messages {
		if m.Role == RoleSystem {
			return m.Content
		}
	}
	return ""
}

// EstimateTokens calculates an approximate token count based on characters, words, and structural overhead.
func (r *ChatCompletionRequest) EstimateTokens() int {
	totalChars := 0
	for _, m := range r.Messages {
		totalChars += len(m.Role) + len(m.Content) + 8 // 8 tokens overhead for message frame
	}
	// Standard BPE rule of thumb: ~4 characters per token
	estimated := totalChars / 4
	if estimated < 4 {
		estimated = 4
	}
	// Add estimated max tokens if set, bounded to prevent arithmetic overflow
	if r.MaxTokens != nil && *r.MaxTokens > 0 {
		maxTok := *r.MaxTokens
		if maxTok > 1000000 {
			maxTok = 1000000
		}
		estimated += maxTok / 4
	}
	return estimated
}

// ChatChoice is a completion choice in a non-streaming response.
type ChatChoice struct {
	Index        int         `json:"index"`
	Message      ChatMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

// Usage captures token counts for the request and response.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// ChatCompletionResponse is the standard OpenAI non-streaming response payload.
type ChatCompletionResponse struct {
	ID                string       `json:"id"`
	Object            string       `json:"object"`
	Created           int64        `json:"created"`
	Model             string       `json:"model"`
	Choices           []ChatChoice `json:"choices"`
	Usage             Usage        `json:"usage"`
	SystemFingerprint string       `json:"system_fingerprint,omitempty"`
	Cached            bool         `json:"x_cached,omitempty"`
	RoutingTier       int          `json:"x_routing_tier,omitempty"`
	SelectedProvider  string       `json:"x_selected_provider,omitempty"`
}

// Delta is the streaming token delta in an SSE chunk.
type Delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// StreamChoice is a completion choice in an SSE chunk.
type StreamChoice struct {
	Index        int    `json:"index"`
	Delta        Delta  `json:"delta"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// StreamChunk is an OpenAI-compatible SSE chunk payload.
type StreamChunk struct {
	ID                string         `json:"id"`
	Object            string         `json:"object"`
	Created           int64          `json:"created"`
	Model             string         `json:"model"`
	Choices           []StreamChoice `json:"choices"`
	SystemFingerprint string         `json:"system_fingerprint,omitempty"`
}

// NewChatCompletionResponse creates a fresh response object.
func NewChatCompletionResponse(model, content, finishReason string, promptTokens, completionTokens int) *ChatCompletionResponse {
	return &ChatCompletionResponse{
		ID:      "chatcmpl-" + uuid.New().String(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: []ChatChoice{
			{
				Index: 0,
				Message: ChatMessage{
					Role:    RoleAssistant,
					Content: content,
				},
				FinishReason: finishReason,
			},
		},
		Usage: Usage{
			PromptTokens:     promptTokens,
			CompletionTokens: completionTokens,
			TotalTokens:      promptTokens + completionTokens,
		},
	}
}

// FormatSSEChunk formats a StreamChunk into the standard SSE line "data: {...}\n\n".
func FormatSSEChunk(chunk *StreamChunk) []byte {
	b, _ := json.Marshal(chunk)
	return []byte(fmt.Sprintf("data: %s\n\n", string(b)))
}

// DoneSSEChunk returns the terminating SSE line "data: [DONE]\n\n".
func DoneSSEChunk() []byte {
	return []byte("data: [DONE]\n\n")
}
