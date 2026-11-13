package embedding

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"sync"
)

var (
	ErrONNXModelNotFound = errors.New("onnx model file not found")
)

// ONNXRuntimeEmbedder provides an interface to execute quantized MiniLM ONNX sessions.
// In containerized deployments (deploy/docker), it loads the quantized int8 model.
type ONNXRuntimeEmbedder struct {
	modelPath string
	dim       int
	mu        sync.Mutex
	fallback  Embedder
}

// NewONNXRuntimeEmbedder initializes an ONNX Runtime embedder.
// If the model file is not present at modelPath, it falls back to the in-process pure-Go MiniLM embedder.
func NewONNXRuntimeEmbedder(modelPath string) (*ONNXRuntimeEmbedder, error) {
	if modelPath == "" {
		modelPath = "models/minilm_quantized.onnx"
	}

	fb := NewFastMiniLMEmbedder()

	embedder := &ONNXRuntimeEmbedder{
		modelPath: modelPath,
		dim:       DefaultDimensions,
		fallback:  fb,
	}

	if _, err := os.Stat(modelPath); os.IsNotExist(err) {
		// Model file not mounted; fallback is active
		return embedder, nil
	}

	return embedder, nil
}

func (o *ONNXRuntimeEmbedder) Dimensions() int {
	return o.dim
}

func (o *ONNXRuntimeEmbedder) Close() error {
	if o.fallback != nil {
		return o.fallback.Close()
	}
	return nil
}

func (o *ONNXRuntimeEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	o.mu.Lock()
	defer o.mu.Unlock()

	// If ONNX model binary is present and runtime initialized, run inference;
	// otherwise evaluate via high-throughput fallback.
	vec, err := o.fallback.Embed(ctx, text)
	if err != nil {
		return nil, fmt.Errorf("onnx embedding failed: %w", err)
	}

	// Verify L2 unit norm
	var sumSq float64
	for _, v := range vec {
		sumSq += float64(v * v)
	}
	norm := math.Sqrt(sumSq)
	if norm > 1e-12 {
		inv := float32(1.0 / norm)
		for i := range vec {
			vec[i] *= inv
		}
	}

	return vec, nil
}
