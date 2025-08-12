# MLflow Model Serving - Makefile
# Automation for development, testing, and deployment

.PHONY: help install install-dev clean lint test test-unit test-integration test-api test-coverage build run run-dev docker-build docker-run docker-compose-up docker-compose-down k8s-deploy k8s-undeploy health-check download-model serve-model benchmark docs format type-check security-check

# Default target
.DEFAULT_GOAL := help

# Configuration
PYTHON := python3
PIP := pip3
DOCKER_IMAGE := mlflow-serving
DOCKER_TAG := latest
NAMESPACE := mlflow-serving
PORT := 8000

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "$(BLUE)MLflow Model Serving - Available Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Usage examples:$(NC)"
	@echo "  make install-dev    # Setup development environment"
	@echo "  make test           # Run all tests"
	@echo "  make docker-build   # Build Docker image"
	@echo "  make k8s-deploy     # Deploy to Kubernetes"

# Development setup
install: ## Install production dependencies
	@echo "$(BLUE)Installing production dependencies...$(NC)"
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies and setup pre-commit
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements-dev.txt
	@if command -v pre-commit >/dev/null 2>&1; then \
		echo "$(BLUE)Setting up pre-commit hooks...$(NC)"; \
		pre-commit install; \
	fi
	@echo "$(GREEN)✅ Development environment ready!$(NC)"

clean: ## Clean up temporary files and caches
	@echo "$(BLUE)Cleaning up...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -name ".coverage" -delete
	find . -name "coverage.xml" -delete
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "$(GREEN)✅ Cleanup complete!$(NC)"

# Code quality
format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black app/ tests/ scripts/
	isort app/ tests/ scripts/
	@echo "$(GREEN)✅ Code formatted!$(NC)"

lint: ## Run linting with flake8
	@echo "$(BLUE)Running linting...$(NC)"
	flake8 app/ tests/ scripts/
	@echo "$(GREEN)✅ Linting passed!$(NC)"

type-check: ## Run type checking with mypy
	@echo "$(BLUE)Running type checks...$(NC)"
	mypy app/
	@echo "$(GREEN)✅ Type checking passed!$(NC)"

security-check: ## Run security checks with bandit
	@echo "$(BLUE)Running security checks...$(NC)"
	@if command -v bandit >/dev/null 2>&1; then \
		bandit -r app/ -f json -o bandit-report.json; \
		echo "$(GREEN)✅ Security check complete! Report: bandit-report.json$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  Bandit not installed. Run: pip install bandit$(NC)"; \
	fi

# Testing
test: test-unit test-integration test-api ## Run all tests

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest tests/ -m unit -v

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest tests/ -m integration -v

test-api: ## Run API tests only
	@echo "$(BLUE)Running API tests...$(NC)"
	pytest tests/ -m api -v

test-coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	pytest tests/ --cov=app --cov-report=html --cov-report=xml --cov-report=term-missing
	@echo "$(GREEN)✅ Coverage report generated: htmlcov/index.html$(NC)"

test-fast: ## Run tests skipping slow ones
	@echo "$(BLUE)Running fast tests...$(NC)"
	pytest tests/ -m "not slow" -v

# Docker operations
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -f docker/Dockerfile -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo "$(GREEN)✅ Docker image built: $(DOCKER_IMAGE):$(DOCKER_TAG)$(NC)"

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -d \
		--name mlflow-serving \
		-p $(PORT):8000 \
		--env-file .env \
		$(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo "$(GREEN)✅ Container running at http://localhost:$(PORT)$(NC)"

docker-stop: ## Stop Docker container
	@echo "$(BLUE)Stopping Docker container...$(NC)"
	docker stop mlflow-serving || true
	docker rm mlflow-serving || true
	@echo "$(GREEN)✅ Container stopped$(NC)"

docker-logs: ## Show Docker container logs
	docker logs -f mlflow-serving

# Docker Compose operations
docker-compose-up: ## Start services with Docker Compose
	@echo "$(BLUE)Starting services with Docker Compose...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)✅ Services started! Check: docker-compose ps$(NC)"

docker-compose-dev: ## Start development services with Docker Compose
	@echo "$(BLUE)Starting development services...$(NC)"
	docker-compose --profile dev up -d
	@echo "$(GREEN)✅ Development services started!$(NC)"

docker-compose-full: ## Start full stack with Docker Compose
	@echo "$(BLUE)Starting full stack (with MLflow and monitoring)...$(NC)"
	docker-compose --profile with-mlflow --profile monitoring up -d
	@echo "$(GREEN)✅ Full stack started!$(NC)"

docker-compose-down: ## Stop Docker Compose services
	@echo "$(BLUE)Stopping Docker Compose services...$(NC)"
	docker-compose down -v
	@echo "$(GREEN)✅ Services stopped$(NC)"

docker-compose-logs: ## Show Docker Compose logs
	docker-compose logs -f

# Kubernetes operations
k8s-deploy: ## Deploy to Kubernetes
	@echo "$(BLUE)Deploying to Kubernetes...$(NC)"
	kubectl apply -f k8s/
	@echo "$(GREEN)✅ Deployed to Kubernetes namespace: $(NAMESPACE)$(NC)"
	@echo "$(YELLOW)Check status: kubectl get pods -n $(NAMESPACE)$(NC)"

k8s-deploy-kustomize: ## Deploy to Kubernetes using Kustomize
	@echo "$(BLUE)Deploying with Kustomize...$(NC)"
	kubectl apply -k k8s/
	@echo "$(GREEN)✅ Deployed with Kustomize$(NC)"

k8s-undeploy: ## Remove from Kubernetes
	@echo "$(BLUE)Removing from Kubernetes...$(NC)"
	kubectl delete -f k8s/ || true
	@echo "$(GREEN)✅ Removed from Kubernetes$(NC)"

k8s-status: ## Check Kubernetes deployment status
	@echo "$(BLUE)Kubernetes deployment status:$(NC)"
	kubectl get pods,services,ingress -n $(NAMESPACE)

k8s-logs: ## Show Kubernetes pod logs
	kubectl logs -f deployment/mlflow-serving -n $(NAMESPACE)

k8s-port-forward: ## Port forward Kubernetes service
	@echo "$(BLUE)Port forwarding to localhost:$(PORT)...$(NC)"
	kubectl port-forward service/mlflow-serving $(PORT):80 -n $(NAMESPACE)

# Application operations
run: ## Run application locally
	@echo "$(BLUE)Starting application...$(NC)"
	$(PYTHON) scripts/start_server.py --host 0.0.0.0 --port $(PORT)

run-dev: ## Run application in development mode
	@echo "$(BLUE)Starting application in development mode...$(NC)"
	$(PYTHON) scripts/start_server.py --host 0.0.0.0 --port $(PORT) --reload

download-model: ## Download model artifacts from S3
	@echo "$(BLUE)Downloading model artifacts...$(NC)"
	$(PYTHON) scripts/download_artifacts.py
	@echo "$(GREEN)✅ Model artifacts downloaded$(NC)"

health-check: ## Run health check
	@echo "$(BLUE)Running health check...$(NC)"
	$(PYTHON) scripts/health_check.py --url http://localhost:$(PORT)

health-check-docker: ## Run health check against Docker container
	@echo "$(BLUE)Running health check against Docker container...$(NC)"
	$(PYTHON) scripts/health_check.py --url http://localhost:$(PORT)

health-check-k8s: ## Run health check against Kubernetes service
	@echo "$(BLUE)Running health check against Kubernetes service...$(NC)"
	kubectl port-forward service/mlflow-serving $(PORT):80 -n $(NAMESPACE) &
	sleep 5
	$(PYTHON) scripts/health_check.py --url http://localhost:$(PORT)
	pkill -f "kubectl port-forward" || true

# Performance and benchmarking
benchmark: ## Run performance benchmark
	@echo "$(BLUE)Running performance benchmark...$(NC)"
	@if command -v locust >/dev/null 2>&1; then \
		echo "Starting Locust benchmark..."; \
		locust -f tests/load_test.py --host http://localhost:$(PORT) --headless -u 10 -r 2 -t 60s; \
	else \
		echo "$(YELLOW)⚠️  Locust not installed. Run: pip install locust$(NC)"; \
		echo "$(BLUE)Running simple benchmark instead...$(NC)"; \
		$(PYTHON) -c "import requests; import time; start=time.time(); [requests.post('http://localhost:$(PORT)/api/v1/predict', json={'question': 'test'}) for _ in range(100)]; print(f'100 requests in {time.time()-start:.2f}s')"; \
	fi

load-test: ## Run load test with Locust
	@echo "$(BLUE)Starting Locust load test...$(NC)"
	@if command -v locust >/dev/null 2>&1; then \
		locust -f tests/load_test.py --host http://localhost:$(PORT); \
	else \
		echo "$(RED)❌ Locust not installed. Run: pip install locust$(NC)"; \
	fi

# Documentation
docs: ## Generate and serve documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@if command -v mkdocs >/dev/null 2>&1; then \
		mkdocs serve; \
	else \
		echo "$(YELLOW)⚠️  MkDocs not installed.$(NC)"; \
		echo "$(BLUE)Opening API docs in browser...$(NC)"; \
		$(PYTHON) -c "import webbrowser; webbrowser.open('http://localhost:$(PORT)/docs')"; \
	fi

docs-build: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	@if command -v mkdocs >/dev/null 2>&1; then \
		mkdocs build; \
		echo "$(GREEN)✅ Documentation built: site/index.html$(NC)"; \
	else \
		echo "$(YELLOW)⚠️  MkDocs not installed. Install with: pip install mkdocs$(NC)"; \
	fi

# Maintenance
upgrade-deps: ## Upgrade dependencies to latest versions
	@echo "$(BLUE)Upgrading dependencies...$(NC)"
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -r requirements.txt
	$(PIP) list --outdated
	@echo "$(GREEN)✅ Dependencies upgraded$(NC)"

check-deps: ## Check for security vulnerabilities in dependencies
	@echo "$(BLUE)Checking dependencies for vulnerabilities...$(NC)"
	@if command -v safety >/dev/null 2>&1; then \
		safety check; \
	else \
		echo "$(YELLOW)⚠️  Safety not installed. Run: pip install safety$(NC)"; \
	fi

# CI/CD helpers
ci-install: ## Install dependencies for CI
	$(PIP) install -r requirements-dev.txt

ci-test: ## Run tests for CI
	pytest tests/ --cov=app --cov-report=xml --cov-report=term-missing --junitxml=test-results.xml

ci-lint: ## Run linting for CI
	flake8 app/ tests/ scripts/ --output-file=flake8-report.txt

ci-security: ## Run security checks for CI
	bandit -r app/ -f json -o bandit-report.json

ci-build: ## Build artifacts for CI
	$(MAKE) docker-build
	docker save $(DOCKER_IMAGE):$(DOCKER_TAG) -o $(DOCKER_IMAGE)-$(DOCKER_TAG).tar

# Environment setup
setup-env: ## Setup environment file from template
	@if [ ! -f .env ]; then \
		echo "$(BLUE)Creating .env file from template...$(NC)"; \
		cp .env.example .env; \
		echo "$(YELLOW)⚠️  Please edit .env file with your configuration$(NC)"; \
	else \
		echo "$(GREEN)✅ .env file already exists$(NC)"; \
	fi

validate-env: ## Validate environment configuration
	@echo "$(BLUE)Validating environment configuration...$(NC)"
	$(PYTHON) -c "from app.core.config import settings; print('✅ Configuration valid')"

# Quick start
quickstart: setup-env install-dev docker-compose-up ## Quick setup for new developers
	@echo "$(GREEN)🚀 Quick start complete!$(NC)"
	@echo ""
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  1. Edit .env file with your configuration"
	@echo "  2. Run: make health-check"
	@echo "  3. Visit: http://localhost:$(PORT)/docs"
	@echo ""
	@echo "$(YELLOW)Available endpoints:$(NC)"
	@echo "  - API Docs: http://localhost:$(PORT)/docs"
	@echo "  - Health: http://localhost:$(PORT)/api/v1/health"
	@echo "  - Metrics: http://localhost:$(PORT)/api/v1/metrics"

# Model Export for Remote Deployment
export-model: ## Export model to standalone Docker package for remote deployment
	@echo "$(BLUE)📤 Exporting model for standalone deployment...$(NC)"
	@echo ""
	@echo "$(YELLOW)Usage:$(NC)"
	@echo "  python scripts/export_model_docker.py \\"
	@echo "    --model-uri models:/your-model/1 \\"
	@echo "    --s3-endpoint-url http://localhost:9878 \\"
	@echo "    --s3-access-key hadoop \\"
	@echo "    --s3-secret-key hadoop \\"
	@echo "    --s3-bucket-name aiongenbucket \\"
	@echo "    --output-dir ./exported_model"
	@echo ""
	@echo "$(GREEN)✨ This creates a completely standalone deployment package!$(NC)"

export-example: ## Show example export command
	@echo "$(BLUE)📝 Example export command:$(NC)"
	@echo ""
	@echo "$(YELLOW)python scripts/export_model_docker.py \\$(NC)"
	@echo "$(YELLOW)  --model-uri models:/langchain-e2e-model/1 \\$(NC)"
	@echo "$(YELLOW)  --s3-endpoint-url http://localhost:9878 \\$(NC)"
	@echo "$(YELLOW)  --s3-access-key hadoop \\$(NC)"
	@echo "$(YELLOW)  --s3-secret-key hadoop \\$(NC)"
	@echo "$(YELLOW)  --s3-bucket-name aiongenbucket \\$(NC)"
	@echo "$(YELLOW)  --output-dir ./exported_model$(NC)"
	@echo ""
	@echo "$(BLUE)Then deploy on any Docker-capable machine:$(NC)"
	@echo "  cd exported_model"
	@echo "  ./build.sh"
	@echo "  ./run.sh"
	@echo "  # Access Swagger UI: http://localhost:8000/docs"

export-and-test: ## Export model and test the package locally
	@echo "$(BLUE)🧪 Exporting and testing model package...$(NC)"
	@echo ""
	@echo "$(YELLOW)⚠️  This requires S3 credentials. Example:$(NC)"
	@echo ""
	@echo "python scripts/export_model_docker.py \\"
	@echo "  --model-uri models:/langchain-e2e-model/1 \\"
	@echo "  --s3-endpoint-url http://localhost:9878 \\"
	@echo "  --s3-access-key hadoop \\"
	@echo "  --s3-secret-key hadoop \\"
	@echo "  --s3-bucket-name aiongenbucket \\"
	@echo "  --output-dir ./test_export && \\"
	@echo "cd test_export && \\"
	@echo "./build.sh && \\"
	@echo "./run.sh && \\"
	@echo "sleep 10 && \\"
	@echo "./test.sh"
	@echo ""
	@echo "$(GREEN)💡 Run this command with your actual credentials!$(NC)"

# All-in-one targets
dev-setup: install-dev setup-env ## Complete development setup
	@echo "$(GREEN)🎉 Development environment ready!$(NC)"

deploy-local: docker-build docker-compose-up health-check ## Build and deploy locally
	@echo "$(GREEN)🚀 Local deployment complete!$(NC)"

deploy-k8s: docker-build k8s-deploy ## Build and deploy to Kubernetes
	@echo "$(GREEN)☸️  Kubernetes deployment complete!$(NC)"

# Status and monitoring
status: ## Show status of all services
	@echo "$(BLUE)=== Service Status ===$(NC)"
	@echo ""
	@echo "$(BLUE)Docker Containers:$(NC)"
	@docker ps --filter "name=mlflow" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || echo "No Docker containers running"
	@echo ""
	@echo "$(BLUE)Kubernetes Pods:$(NC)"
	@kubectl get pods -n $(NAMESPACE) 2>/dev/null || echo "Not deployed to Kubernetes"
	@echo ""
	@echo "$(BLUE)Health Check:$(NC)"
	@curl -s http://localhost:$(PORT)/api/v1/health 2>/dev/null && echo "✅ Service healthy" || echo "❌ Service not responding"

monitor: ## Monitor logs in real-time
	@echo "$(BLUE)Monitoring logs... (Press Ctrl+C to stop)$(NC)"
	@if docker ps --filter "name=mlflow-serving" -q | grep -q .; then \
		docker logs -f mlflow-serving; \
	elif kubectl get pods -n $(NAMESPACE) -l app.kubernetes.io/name=mlflow-serving -o name | grep -q .; then \
		kubectl logs -f deployment/mlflow-serving -n $(NAMESPACE); \
	else \
		echo "$(RED)❌ No running services found$(NC)"; \
	fi