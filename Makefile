.PHONY: install run test lint format clean help

# Default target
help:
	@echo "Poe Bots Framework Makefile"
	@echo ""
	@echo "Usage:"
	@echo "  make install     Install dependencies"
	@echo "  make run         Run the server"
	@echo "  make run-debug   Run the server in debug mode"
	@echo "  make test        Run tests"
	@echo "  make lint        Run linters"
	@echo "  make format      Format code with black"
	@echo "  make clean       Clean up build artifacts"
	@echo "  make help        Show this help"

# Install dependencies
install:
	pip install -r requirements.txt
	pip install pytest black ruff pyright

# Run the server
run:
	./run_local.sh

# Run the server in debug mode
run-debug:
	./run_local.sh --debug --reload

# Run tests
test:
	pytest tests/

# Run linters
lint:
	ruff check .
	pyright .

# Format code
format:
	black .

# Clean up build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +