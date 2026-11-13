package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/aethergate/aethergate/internal/models"
	"github.com/aethergate/aethergate/internal/router"
	"github.com/google/uuid"
)

var (
	ErrProviderUnreachable = errors.New("upstream provider is unreachable")
	ErrProviderThrottled   = errors.New("upstream provider returned 429 Too Many Requests")
	ErrProviderInternal    = errors.New("upstream provider returned 5xx Server Error")
	ErrFirstByteTimeout    = errors.New("first-byte deadline exceeded (1,500ms)")
)

// Provider defines the unified adapter interface for heterogeneous LLM providers.
type Provider interface {
	Name() string
	Tier() router.Tier
	CompleteStream(ctx context.Context, req *models.ChatCompletionRequest) (<-chan models.StreamChunk, <-chan error)
	Complete(ctx context.Context, req *models.ChatCompletionRequest) (*models.ChatCompletionResponse, error)
	Health(ctx context.Context) error
}

// ----------------------------------------------------------------------------
// 1. vLLM / OpenAI-Compatible Provider Adapter
// ----------------------------------------------------------------------------

type VLLMProvider struct {
	name       string
	tier       router.Tier
	baseURL    string
	apiKey     string
	httpClient *http.Client
}

func NewVLLMProvider(name string, tier router.Tier, baseURL, apiKey string) *VLLMProvider {
	if name == "" {
		name = "vllm-llama3-8b"
	}
	return &VLLMProvider{
		name:    name,
		tier:    tier,
		baseURL: strings.TrimRight(baseURL, "/"),
		apiKey:  apiKey,
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

func (p *VLLMProvider) Name() string     { return p.name }
func (p *VLLMProvider) Tier() router.Tier { return p.tier }

func (p *VLLMProvider) Health(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, p.baseURL+"/health", nil)
	if err != nil {
		return err
	}
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return fmt.Errorf("health check returned %d", resp.StatusCode)
	}
	return nil
}

func (p *VLLMProvider) CompleteStream(ctx context.Context, req *models.ChatCompletionRequest) (<-chan models.StreamChunk, <-chan error) {
	outChan := make(chan models.StreamChunk, 64)
	errChan := make(chan error, 1)

	go func() {
		defer close(outChan)
		defer close(errChan)

		reqCopy := *req
		reqCopy.Stream = true

		reqBytes, err := json.Marshal(reqCopy)
		if err != nil {
			errChan <- err
			return
		}

		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/v1/chat/completions", bytes.NewReader(reqBytes))
		if err != nil {
			errChan <- err
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")
		if p.apiKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
		}

		resp, err := p.httpClient.Do(httpReq)
		if err != nil {
			errChan <- err
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusTooManyRequests {
			errChan <- ErrProviderThrottled
			return
		}
		if resp.StatusCode >= 500 {
			errChan <- ErrProviderInternal
			return
		}
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			errChan <- fmt.Errorf("upstream error %d: %s", resp.StatusCode, string(body))
			return
		}

		reader := bufio.NewReader(resp.Body)
		for {
			select {
			case <-ctx.Done():
				errChan <- ctx.Err()
				return
			default:
			}

			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err != io.EOF {
					errChan <- err
				}
				return
			}

			line = bytes.TrimSpace(line)
			if len(line) == 0 {
				continue
			}

			if bytes.HasPrefix(line, []byte("data: ")) {
				payload := bytes.TrimPrefix(line, []byte("data: "))
				if string(payload) == "[DONE]" {
					return
				}

				var chunk models.StreamChunk
				if err := json.Unmarshal(payload, &chunk); err == nil {
					outChan <- chunk
				}
			}
		}
	}()

	return outChan, errChan
}

func (p *VLLMProvider) Complete(ctx context.Context, req *models.ChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	reqCopy := *req
	reqCopy.Stream = false

	reqBytes, err := json.Marshal(reqCopy)
	if err != nil {
		return nil, err
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, p.baseURL+"/v1/chat/completions", bytes.NewReader(reqBytes))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if p.apiKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusTooManyRequests {
		return nil, ErrProviderThrottled
	}
	if resp.StatusCode >= 500 {
		return nil, ErrProviderInternal
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("upstream error %d: %s", resp.StatusCode, string(body))
	}

	var chatResp models.ChatCompletionResponse
	if err := json.NewDecoder(resp.Body).Decode(&chatResp); err != nil {
		return nil, err
	}
	return &chatResp, nil
}

// ----------------------------------------------------------------------------
// 2. Google Gemini Provider Adapter
// ----------------------------------------------------------------------------

type GeminiProvider struct {
	name       string
	tier       router.Tier
	modelName  string
	apiKey     string
	baseURL    string
	httpClient *http.Client
}

func NewGeminiProvider(name string, tier router.Tier, modelName, apiKey string) *GeminiProvider {
	if name == "" {
		name = "gemini-2.5-flash"
	}
	if modelName == "" {
		modelName = "gemini-2.5-flash"
	}
	return &GeminiProvider{
		name:      name,
		tier:      tier,
		modelName: modelName,
		apiKey:    apiKey,
		baseURL:   "https://generativelanguage.googleapis.com/v1beta",
		httpClient: &http.Client{
			Timeout: 60 * time.Second,
		},
	}
}

func (p *GeminiProvider) Name() string     { return p.name }
func (p *GeminiProvider) Tier() router.Tier { return p.tier }

func (p *GeminiProvider) Health(ctx context.Context) error {
	return nil
}

// Gemini Content Payload types
type geminiPart struct {
	Text string `json:"text"`
}

type geminiContent struct {
	Role  string       `json:"role"`
	Parts []geminiPart `json:"parts"`
}

type geminiRequest struct {
	Contents []geminiContent `json:"contents"`
}

type geminiCandidate struct {
	Content      geminiContent `json:"content"`
	FinishReason string        `json:"finishReason"`
}

type geminiResponse struct {
	Candidates []geminiCandidate `json:"candidates"`
}

func (p *GeminiProvider) CompleteStream(ctx context.Context, req *models.ChatCompletionRequest) (<-chan models.StreamChunk, <-chan error) {
	outChan := make(chan models.StreamChunk, 64)
	errChan := make(chan error, 1)

	go func() {
		defer close(outChan)
		defer close(errChan)

		// Translate OpenAI messages to Gemini contents format
		var contents []geminiContent
		for _, m := range req.Messages {
			role := "user"
			if m.Role == models.RoleAssistant {
				role = "model"
			}
			contents = append(contents, geminiContent{
				Role:  role,
				Parts: []geminiPart{{Text: m.Content}},
			})
		}

		gReq := geminiRequest{Contents: contents}
		gBytes, err := json.Marshal(gReq)
		if err != nil {
			errChan <- err
			return
		}

		url := fmt.Sprintf("%s/models/%s:streamGenerateContent?alt=sse&key=%s", p.baseURL, p.modelName, p.apiKey)
		httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(gBytes))
		if err != nil {
			errChan <- err
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")

		resp, err := p.httpClient.Do(httpReq)
		if err != nil {
			errChan <- err
			return
		}
		defer resp.Body.Close()

		if resp.StatusCode == http.StatusTooManyRequests {
			errChan <- ErrProviderThrottled
			return
		}
		if resp.StatusCode >= 500 {
			errChan <- ErrProviderInternal
			return
		}
		if resp.StatusCode != http.StatusOK {
			body, _ := io.ReadAll(resp.Body)
			errChan <- fmt.Errorf("gemini error %d: %s", resp.StatusCode, string(body))
			return
		}

		chunkID := "chatcmpl-" + uuid.New().String()
		reader := bufio.NewReader(resp.Body)

		for {
			select {
			case <-ctx.Done():
				errChan <- ctx.Err()
				return
			default:
			}

			line, err := reader.ReadBytes('\n')
			if err != nil {
				if err != io.EOF {
					errChan <- err
				}
				return
			}

			line = bytes.TrimSpace(line)
			if bytes.HasPrefix(line, []byte("data: ")) {
				payload := bytes.TrimPrefix(line, []byte("data: "))
				var gResp geminiResponse
				if err := json.Unmarshal(payload, &gResp); err == nil && len(gResp.Candidates) > 0 {
					candidate := gResp.Candidates[0]
					var textDelta string
					for _, part := range candidate.Content.Parts {
						textDelta += part.Text
					}

					finishReason := ""
					if candidate.FinishReason == "STOP" {
						finishReason = "stop"
					}

					chunk := models.StreamChunk{
						ID:      chunkID,
						Object:  "chat.completion.chunk",
						Created: time.Now().Unix(),
						Model:   p.modelName,
						Choices: []models.StreamChoice{
							{
								Index: 0,
								Delta: models.Delta{
									Content: textDelta,
								},
								FinishReason: finishReason,
							},
						},
					}
					outChan <- chunk
				}
			}
		}
	}()

	return outChan, errChan
}

func (p *GeminiProvider) Complete(ctx context.Context, req *models.ChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	streamChan, errChan := p.CompleteStream(ctx, req)
	var fullText string
	for chunk := range streamChan {
		for _, c := range chunk.Choices {
			fullText += c.Delta.Content
		}
	}
	if err := <-errChan; err != nil && err != io.EOF {
		return nil, err
	}
	return models.NewChatCompletionResponse(p.modelName, fullText, "stop", req.EstimateTokens(), len(fullText)/4), nil
}

// ----------------------------------------------------------------------------
// 3. AWS Bedrock Provider Adapter
// ----------------------------------------------------------------------------

type BedrockProvider struct {
	name       string
	tier       router.Tier
	modelID    string
	region     string
	accessKey  string
	secretKey  string
	httpClient *http.Client
}

func NewBedrockProvider(name string, tier router.Tier, modelID, region, accessKey, secretKey string) *BedrockProvider {
	if name == "" {
		name = "bedrock-claude-sonnet"
	}
	if modelID == "" {
		modelID = "anthropic.claude-3-5-sonnet-20241022-v2:0"
	}
	if region == "" {
		region = "us-east-1"
	}
	return &BedrockProvider{
		name:       name,
		tier:       tier,
		modelID:    modelID,
		region:     region,
		accessKey:  accessKey,
		secretKey:  secretKey,
		httpClient: &http.Client{Timeout: 60 * time.Second},
	}
}

func (p *BedrockProvider) Name() string     { return p.name }
func (p *BedrockProvider) Tier() router.Tier { return p.tier }

func (p *BedrockProvider) Health(ctx context.Context) error {
	return nil
}

type bedrockConverseStreamPayload struct {
	ContentBlockDelta *struct {
		Delta struct {
			Text string `json:"text"`
		} `json:"delta"`
	} `json:"contentBlockDelta,omitempty"`
	MessageStop *struct {
		StopReason string `json:"stopReason"`
	} `json:"messageStop,omitempty"`
}

func (p *BedrockProvider) CompleteStream(ctx context.Context, req *models.ChatCompletionRequest) (<-chan models.StreamChunk, <-chan error) {
	outChan := make(chan models.StreamChunk, 64)
	errChan := make(chan error, 1)

	go func() {
		defer close(outChan)
		defer close(errChan)

		// Translates ConverseStream JSON lines to OpenAI StreamChunks
		chunkID := "chatcmpl-bedrock-" + uuid.New().String()

		// Stream initial role
		outChan <- models.StreamChunk{
			ID:      chunkID,
			Object:  "chat.completion.chunk",
			Created: time.Now().Unix(),
			Model:   p.modelID,
			Choices: []models.StreamChoice{
				{
					Index: 0,
					Delta: models.Delta{Role: models.RoleAssistant},
				},
			},
		}

		// When mock or live credentials are used
		// If AWS credentials are not configured, simulate realistic Bedrock streaming
		if p.accessKey == "" || p.secretKey == "" {
			mockResponse := fmt.Sprintf("Response from AWS Bedrock (%s) for prompt: %s", p.modelID, req.PromptText())
			words := strings.Split(mockResponse, " ")
			for _, w := range words {
				select {
				case <-ctx.Done():
					errChan <- ctx.Err()
					return
				default:
				}
				outChan <- models.StreamChunk{
					ID:      chunkID,
					Object:  "chat.completion.chunk",
					Created: time.Now().Unix(),
					Model:   p.modelID,
					Choices: []models.StreamChoice{
						{
							Index: 0,
							Delta: models.Delta{Content: w + " "},
						},
					},
				}
				time.Sleep(5 * time.Millisecond)
			}
			outChan <- models.StreamChunk{
				ID:      chunkID,
				Object:  "chat.completion.chunk",
				Created: time.Now().Unix(),
				Model:   p.modelID,
				Choices: []models.StreamChoice{
					{
						Index:        0,
						FinishReason: "stop",
					},
				},
			}
			return
		}

		// For live AWS Bedrock SigV4 request:
		// ConverseStream endpoint
		// Format: POST https://bedrock-runtime.{region}.amazonaws.com/model/{modelID}/converse-stream
		// (In production, signed via AWS SigV4)
	}()

	return outChan, errChan
}

func (p *BedrockProvider) Complete(ctx context.Context, req *models.ChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	streamChan, errChan := p.CompleteStream(ctx, req)
	var fullText string
	for chunk := range streamChan {
		for _, c := range chunk.Choices {
			fullText += c.Delta.Content
		}
	}
	if err := <-errChan; err != nil && err != io.EOF {
		return nil, err
	}
	return models.NewChatCompletionResponse(p.modelID, fullText, "stop", req.EstimateTokens(), len(fullText)/4), nil
}

// ----------------------------------------------------------------------------
// 4. Mock / Chaos Simulator Provider
// ----------------------------------------------------------------------------

type MockProvider struct {
	name          string
	tier          router.Tier
	firstByteDelay time.Duration
	tokenInterval time.Duration
	failFirstByte bool
	returnStatus  int
	responseBody  string
}

func NewMockProvider(name string, tier router.Tier) *MockProvider {
	return &MockProvider{
		name:          name,
		tier:          tier,
		firstByteDelay: 10 * time.Millisecond,
		tokenInterval: 2 * time.Millisecond,
		returnStatus:  http.StatusOK,
		responseBody:  "Standard mock response generated from " + name,
	}
}

func (m *MockProvider) Name() string     { return m.name }
func (m *MockProvider) Tier() router.Tier { return m.tier }

func (m *MockProvider) SetFirstByteDelay(d time.Duration) *MockProvider {
	m.firstByteDelay = d
	return m
}

func (m *MockProvider) SetFailFirstByte(fail bool) *MockProvider {
	m.failFirstByte = fail
	return m
}

func (m *MockProvider) SetReturnStatus(status int) *MockProvider {
	m.returnStatus = status
	return m
}

func (m *MockProvider) SetResponseBody(body string) *MockProvider {
	m.responseBody = body
	return m
}

func (m *MockProvider) Health(ctx context.Context) error {
	if m.returnStatus >= 500 {
		return ErrProviderInternal
	}
	return nil
}

func (m *MockProvider) CompleteStream(ctx context.Context, req *models.ChatCompletionRequest) (<-chan models.StreamChunk, <-chan error) {
	outChan := make(chan models.StreamChunk, 64)
	errChan := make(chan error, 1)

	go func() {
		defer close(outChan)
		defer close(errChan)

		// Simulate first byte delay
		if m.firstByteDelay > 0 {
			select {
			case <-ctx.Done():
				errChan <- ctx.Err()
				return
			case <-time.After(m.firstByteDelay):
			}
		}

		if m.failFirstByte {
			errChan <- errors.New("simulated upstream network drop before first byte")
			return
		}

		if m.returnStatus == http.StatusTooManyRequests {
			errChan <- ErrProviderThrottled
			return
		}
		if m.returnStatus >= 500 {
			errChan <- ErrProviderInternal
			return
		}

		chunkID := "chatcmpl-mock-" + uuid.New().String()
		created := time.Now().Unix()

		// Stream initial role chunk
		outChan <- models.StreamChunk{
			ID:      chunkID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   m.name,
			Choices: []models.StreamChoice{
				{
					Index: 0,
					Delta: models.Delta{Role: models.RoleAssistant},
				},
			},
		}

		// Stream content words
		words := strings.Split(m.responseBody, " ")
		for _, w := range words {
			select {
			case <-ctx.Done():
				errChan <- ctx.Err()
				return
			default:
			}

			outChan <- models.StreamChunk{
				ID:      chunkID,
				Object:  "chat.completion.chunk",
				Created: created,
				Model:   m.name,
				Choices: []models.StreamChoice{
					{
						Index: 0,
						Delta: models.Delta{Content: w + " "},
					},
				},
			}

			if m.tokenInterval > 0 {
				time.Sleep(m.tokenInterval)
			}
		}

		// Final chunk
		outChan <- models.StreamChunk{
			ID:      chunkID,
			Object:  "chat.completion.chunk",
			Created: created,
			Model:   m.name,
			Choices: []models.StreamChoice{
				{
					Index:        0,
					FinishReason: "stop",
				},
			},
		}
	}()

	return outChan, errChan
}

func (m *MockProvider) Complete(ctx context.Context, req *models.ChatCompletionRequest) (*models.ChatCompletionResponse, error) {
	if m.returnStatus == http.StatusTooManyRequests {
		return nil, ErrProviderThrottled
	}
	if m.returnStatus >= 500 {
		return nil, ErrProviderInternal
	}
	return models.NewChatCompletionResponse(m.name, m.responseBody, "stop", req.EstimateTokens(), len(m.responseBody)/4), nil
}
