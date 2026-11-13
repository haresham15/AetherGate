package proxy

import (
	"net/http"
	"strings"
	"testing"
)

func TestSignAWSBedrockRequest(t *testing.T) {
	req, err := http.NewRequest(http.MethodPost, "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-3-5-sonnet/converse-stream", nil)
	if err != nil {
		t.Fatalf("failed to create request: %v", err)
	}

	body := []byte(`{"messages":[{"role":"user","content":[{"text":"Hello"}]}]}`)
	accessKey := "AKIAIOSFODNN7EXAMPLE"
	secretKey := "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
	region := "us-east-1"

	err = SignAWSBedrockRequest(req, body, region, accessKey, secretKey, "")
	if err != nil {
		t.Fatalf("SignAWSBedrockRequest failed: %v", err)
	}

	authHeader := req.Header.Get("Authorization")
	if !strings.HasPrefix(authHeader, "AWS4-HMAC-SHA256 Credential=AKIAIOSFODNN7EXAMPLE/") {
		t.Fatalf("invalid authorization header: %s", authHeader)
	}
	if !strings.Contains(authHeader, "SignedHeaders=host;x-amz-content-sha256;x-amz-date") {
		t.Fatalf("expected signed headers in auth header: %s", authHeader)
	}
	if !strings.Contains(authHeader, "Signature=") {
		t.Fatalf("expected signature in auth header: %s", authHeader)
	}

	amzDate := req.Header.Get("x-amz-date")
	if amzDate == "" {
		t.Fatalf("expected x-amz-date header to be set")
	}

	contentSha := req.Header.Get("x-amz-content-sha256")
	if contentSha == "" {
		t.Fatalf("expected x-amz-content-sha256 to be set")
	}
}
