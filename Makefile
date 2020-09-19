dev:
	pip install -U -r requirements.txt
	pre-commit install
	sh apply.sh

format:
	black .
	isort .

test:
	black . --check
	isort . --check
	pytest --cov-report=xml --cov=./ tests/*