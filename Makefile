HOOK_IMAGE_NAME ?= mlops-app-stable-diffusion
HOOK_IMAGE_TAG ?= latest

.PHONY: all clean test lint format
all clean test lint format:

SHELL := /bin/sh -e

.PHONY: install
install:
	poetry config virtualenvs.in-project true
	poetry install --with dev
	poetry run pre-commit install;


.PHONY: lint format
lint format:
ifdef CI
	poetry run pre-commit run --all-files --show-diff-on-failure
	poetry run mypy --strict .apolo/src
else
	# automatically fix the formatting issues and rerun again
	poetry run pre-commit run --all-files || pre-commit run --all-files
	poetry run mypy --strict .apolo/src
endif

.PHONY: test
test:

.PHONY: clean
clean:

.PHONY: test-unit
test-unit:
	poetry run pytest -vvs --cov=.apolo --cov-report xml:.coverage.unit.xml .apolo/tests/unit


.PHONY: build-hook-image
build-hook-image:
	docker build \
		--build-arg APP_IMAGE_TAG=$(HOOK_IMAGE_TAG) \
		-t $(HOOK_IMAGE_NAME):latest \
		-f hooks.Dockerfile \
		.;

.PHONY: push-hook-image
push-hook-image:
	docker tag $(HOOK_IMAGE_NAME):latest ghcr.io/neuro-inc/$(HOOK_IMAGE_NAME):$(HOOK_IMAGE_TAG)
	docker push ghcr.io/neuro-inc/$(HOOK_IMAGE_NAME):$(HOOK_IMAGE_TAG)

.PHONY: gen-types-schemas
gen-types-schemas:
	app-types dump-types-schema .apolo/src/apolo_apps_stable_diffusion StableDiffusionInputs .apolo/src/apolo_apps_stable_diffusion/schemas/StableDiffusionInputs.json
	app-types dump-types-schema .apolo/src/apolo_apps_stable_diffusion StableDiffusionOutputs .apolo/src/apolo_apps_stable_diffusion/schemas/StableDiffusionOutputs.json
