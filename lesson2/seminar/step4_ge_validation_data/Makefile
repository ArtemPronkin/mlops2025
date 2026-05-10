install:
	poetry install

pre-commit:
	poetry run pre-commit run --all-files

repro:
	poetry run dvc repro

metrics:
	poetry run dvc metrics show

clean:
	rm -rf data/processed models metrics reports
