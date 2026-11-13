.PHONY: all build test bench clean docker-up docker-down lint

BINARY_NAME=aethergate
BENCHMARK_NAME=benchmark

all: test build

build:
	@echo "==> Building AetherGate binary..."
	go build -ldflags="-w -s" -o bin/$(BINARY_NAME) ./cmd/aethergate
	@echo "==> Building Benchmark tool..."
	go build -ldflags="-w -s" -o bin/$(BENCHMARK_NAME) ./cmd/benchmark

test:
	@echo "==> Running full test suite..."
	go test -v -count=1 ./...

test-race:
	@echo "==> Running test suite with race detector..."
	go test -v -race ./...

bench:
	@echo "==> Running synthetic benchmark against localhost:8080..."
	go run ./cmd/benchmark -url http://localhost:8080/v1/chat/completions -n 300 -c 15

docker-up:
	@echo "==> Starting Docker topology (AetherGate + Redis Stack + Mock Upstream + Prometheus + Grafana)..."
	docker compose -f deploy/docker/docker-compose.yml up -d

docker-down:
	@echo "==> Stopping Docker topology..."
	docker compose -f deploy/docker/docker-compose.yml down

clean:
	@echo "==> Cleaning build artifacts..."
	rm -rf bin/ *.exe *.test
