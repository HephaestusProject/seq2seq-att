dev:
	pip install -U -r requirements.txt
	pre-commit install

format:
	black .
	isort .

test:
	black . --check
	isort . --check
	pytest --cov-report=xml --cov=./ tests/*