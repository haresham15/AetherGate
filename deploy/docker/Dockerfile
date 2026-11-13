# Multi-Stage Build for AetherGate Gateway & Benchmark Runner
FROM golang:1.24-alpine AS builder

WORKDIR /app

RUN apk add --no-cache git ca-certificates tzdata

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build AetherGate core gateway
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o /bin/aethergate ./cmd/aethergate

# Build Benchmark & Chaos suite
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o /bin/benchmark ./cmd/benchmark

# Build Mock Upstream Provider
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-w -s" -o /bin/mock-upstream ./deploy/mock_upstream

# Final Minimal Distroless / Alpine Runtime
FROM alpine:3.20

RUN apk add --no-cache ca-certificates tzdata curl

WORKDIR /app

COPY --from=builder /bin/aethergate /usr/local/bin/aethergate
COPY --from=builder /bin/benchmark /usr/local/bin/benchmark
COPY --from=builder /bin/mock-upstream /usr/local/bin/mock-upstream

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/aethergate"]
