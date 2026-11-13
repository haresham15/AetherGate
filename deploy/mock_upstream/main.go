package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/aethergate/aethergate/internal/models"
	"github.com/google/uuid"
)

func main() {
	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"status":"ok"}`))
	})

	mux.HandleFunc("/v1/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req models.ChatCompletionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		prompt := req.PromptText()
		responseContent := fmt.Sprintf("vLLM Local Llama-3-8B response for: %s", prompt)

		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			flusher, _ := w.(http.Flusher)

			chunkID := "chatcmpl-" + uuid.New().String()
			created := time.Now().Unix()

			words := strings.Split(responseContent, " ")
			for _, wStr := range words {
				chunk := models.StreamChunk{
					ID:      chunkID,
					Object:  "chat.completion.chunk",
					Created: created,
					Model:   "vllm-llama3-8b",
					Choices: []models.StreamChoice{
						{
							Index: 0,
							Delta: models.Delta{Content: wStr + " "},
						},
					},
				}
				w.Write(models.FormatSSEChunk(&chunk))
				if flusher != nil {
					flusher.Flush()
				}
				time.Sleep(3 * time.Millisecond)
			}

			w.Write(models.DoneSSEChunk())
			if flusher != nil {
				flusher.Flush()
			}
			return
		}

		resp := models.NewChatCompletionResponse("vllm-llama3-8b", responseContent, "stop", req.EstimateTokens(), len(responseContent)/4)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(resp)
	})

	log.Println("[Mock vLLM] Listening on :8000...")
	if err := http.ListenAndServe(":8000", mux); err != nil {
		log.Fatalf("Mock upstream failed: %v", err)
	}
}
