# Makefile

.PHONY: pre-commit run-server test performance lint

pre-commit:
	@echo "Uninstalling and reinstalling pre-commit hooks..."
	poetry run pre-commit uninstall
	poetry run pre-commit install

run-server:
	@echo "Starting FastAPI application with Uvicorn on port 3000..."
	poetry run uvicorn app.main:app --reload --workers 1 --host 0.0.0.0 --port 3000

test:
	@echo "Running tests for app/tests/..."
	poetry run pytest -v tests/app/

performance:
	@echo "Running performance tests with Locust..."
	poetry run locust -f app/tests/api/endpoints/locust.py --headless -H http://localhost:3000 -u 10 -r 1 -t 40

lint:
	@echo "Running pre-commit hooks for linting and formatting on all files..."
	poetry run pre-commit run --all-files
