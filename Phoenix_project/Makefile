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
	@echo "Running linter..."
	# Run flake8 linter (install with: pip install flake8)
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

build:
	@echo "Building Docker images..."
	docker-compose build

stop:
	@echo "Stopping services..."
	docker-compose down
