# Makefile for Phoenix Project (Layer 15)

.PHONY: run test lint build stop

run:
	@echo "Starting main services (API, worker)..."
	docker-compose up -d --build

test:
	@echo "Running tests..."
	# Run tests with flags from Layer 8
	pytest --maxfail=1 --disable-warnings -q

lint:
	# [Fix III.1] 更新 Linter 为 Ruff
	@echo "Running linter (Ruff)..."
	ruff check .
	ruff format .

build:
	@echo "Building Docker images..."
	docker-compose build

stop:
	@echo "Stopping services..."
	docker-compose down
