package proxy

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"
)

// SignAWSBedrockRequest signs an HTTP request with AWS Signature Version 4 (SigV4)
// for the AWS Bedrock Runtime ConverseStream service.
func SignAWSBedrockRequest(req *http.Request, body []byte, region, accessKey, secretKey, sessionToken string) error {
	if accessKey == "" || secretKey == "" {
		return nil
	}
	if region == "" {
		region = "us-east-1"
	}
	service := "bedrock"

	now := time.Now().UTC()
	dateStamp := now.Format("20060102")
	amzDate := now.Format("20060102T150405Z")

	req.Header.Set("x-amz-date", amzDate)
	if sessionToken != "" {
		req.Header.Set("x-amz-security-token", sessionToken)
	}

	// Payload hash
	payloadHash := sha256Hex(body)
	req.Header.Set("x-amz-content-sha256", payloadHash)

	// Host header
	if req.Header.Get("Host") == "" {
		req.Header.Set("Host", req.URL.Host)
	}

	// 1. Canonical Headers & Signed Headers
	headersToSign := []string{"host", "x-amz-content-sha256", "x-amz-date"}
	if sessionToken != "" {
		headersToSign = append(headersToSign, "x-amz-security-token")
	}
	if ct := req.Header.Get("Content-Type"); ct != "" {
		headersToSign = append(headersToSign, "content-type")
	}
	sort.Strings(headersToSign)

	var canonicalHeaders strings.Builder
	for _, h := range headersToSign {
		val := strings.TrimSpace(req.Header.Get(h))
		if h == "host" && val == "" {
			val = req.URL.Host
		}
		canonicalHeaders.WriteString(fmt.Sprintf("%s:%s\n", strings.ToLower(h), val))
	}
	signedHeaders := strings.Join(headersToSign, ";")

	// 2. Canonical Request
	canonicalURI := req.URL.EscapedPath()
	if canonicalURI == "" {
		canonicalURI = "/"
	}
	canonicalQuery := req.URL.RawQuery

	canonicalRequest := fmt.Sprintf("%s\n%s\n%s\n%s\n%s\n%s",
		req.Method,
		canonicalURI,
		canonicalQuery,
		canonicalHeaders.String(),
		signedHeaders,
		payloadHash,
	)

	// 3. String to Sign
	credentialScope := fmt.Sprintf("%s/%s/%s/aws4_request", dateStamp, region, service)
	stringToSign := fmt.Sprintf("AWS4-HMAC-SHA256\n%s\n%s\n%s",
		amzDate,
		credentialScope,
		sha256Hex([]byte(canonicalRequest)),
	)

	// 4. Calculate Signature
	kSecret := []byte("AWS4" + secretKey)
	kDate := hmacSHA256(kSecret, []byte(dateStamp))
	kRegion := hmacSHA256(kDate, []byte(region))
	kService := hmacSHA256(kRegion, []byte(service))
	kSigning := hmacSHA256(kService, []byte("aws4_request"))
	signature := hex.EncodeToString(hmacSHA256(kSigning, []byte(stringToSign)))

	// 5. Add Authorization Header
	authHeader := fmt.Sprintf("AWS4-HMAC-SHA256 Credential=%s/%s, SignedHeaders=%s, Signature=%s",
		accessKey,
		credentialScope,
		signedHeaders,
		signature,
	)
	req.Header.Set("Authorization", authHeader)
	return nil
}

func sha256Hex(data []byte) string {
	h := sha256.Sum256(data)
	return hex.EncodeToString(h[:])
}

func hmacSHA256(key, data []byte) []byte {
	mac := hmac.New(sha256.New, key)
	mac.Write(data)
	return mac.Sum(nil)
}
