dev:
	pip install -U -r requirements.txt
	pre-commit install
	sh apply.sh

format:
	black .
	isort . --skip wandb

test:
	black . --check
	isort . --check --skip wandb
	pytest --cov-report=xml --cov=./ tests/*