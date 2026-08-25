UID := $(shell id -u)
GID := $(shell id -g)
PWD = $(shell pwd)

args=
# if dockerfile is not defined
ifndef dockerfile
	dockerfile=compose.yml
endif

docker-compose-files=-f ${dockerfile}

help:  ## Show help.
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[$$()% a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

build:  ## Build docker image(s) [args="<name_of_image>"].
	@docker compose $(docker-compose-files) build ${args}

build-no-cache:  ## Build docker image(s) without cache [args="<name_of_image>"].
	@docker compose $(docker-compose-files) --compatibility build ${args} --no-cache

up:  ## Start docker container(s) [args="<name_of_service>"].
	@docker compose ${docker-compose-files} up ${args} -d

down:  ## Stop docker container(s) [args="<name_of_service>"].
	@docker compose ${docker-compose-files} down ${args}

stop:  ## Stop docker containers [args="<name_of_service>"].
	@docker compose ${docker-compose-files} stop ${args}

restart:  ## Restart service(s) [args="<name_of_service>"].
	@docker compose ${docker-compose-files} restart ${args}

test:  ## Run tests.
	@docker exec grinning_cat_core uv run python -m pytest --color=yes -vvv -W ignore --disable-warnings ${args}

install: ## Update the local virtual environment with the latest requirements.
	@uv sync --link-mode=copy --frozen --no-install-project --no-upgrade --no-cache
	@find $(PWD)/cat/core_plugins -name requirements.txt -exec uv pip install --link-mode=copy --no-cache --no-upgrade -r {} \;
	@uv cache clean
	@pip cache purge

update: ## Update and compile requirements for the local virtual environment.
	@uv sync --upgrade --link-mode=copy --no-install-project --no-cache --extra plugins
	@uv cache clean
	@pip cache purge
	@rm -rf *.egg-info

check: ## Check requirements for the local virtual environment.
	@uv sync --check

migrate:  ## Apply database migrations
	@docker exec -it grinning_cat_core uv run python migrations/manage_migrations.py upgrade head

make-migration:  ## Create the migration file after changing the models. Argument `args` is mandatory as the comment of the migration.
	@if [ -z "${args}" ]; then \
		echo "Error: 'args' is required for 'run'. Example: make make-migration args=\"The comment to the migration\"" >&2; \
		exit 1; \
	fi
	@docker exec -it grinning_cat_core uv run python migrations/manage_migrations.py revision -m "${args}"

BUILD_PROXY ?= http://10.42.0.1:3142
BUILD_NO_PROXY ?= localhost,127.0.0.1,::1,.intranet

dhi: update-requirements
	docker buildx build . -f Dockerfile:dhi -t grinning-cat-core:dhi-full \
		--build-arg HTTP_PROXY=$(BUILD_PROXY) \
		--build-arg HTTPS_PROXY=$(BUILD_PROXY) \
		--build-arg NO_PROXY=$(BUILD_NO_PROXY)
dev: update-requirements
	docker buildx build . -f Dockerfile:dhi -t grinning-cat-core:dev-full \
		--build-arg HTTP_PROXY=$(BUILD_PROXY) \
		--build-arg HTTPS_PROXY=$(BUILD_PROXY) \
		--build-arg NO_PROXY=$(BUILD_NO_PROXY)
update-requirements:
	git pull
	docker pull dhi.io/python:3.13-dev
	uv lock --upgrade --python 3.13
	# NOTE: --upgrade is required for `uv pip compile`, otherwise it treats the
	# existing output file's pins as preferences and never tracks the new lock.
	uv pip compile --upgrade --python-version 3.13 --extra plugins -o requirements-full.txt pyproject.toml
	uv pip compile --upgrade -o requirements.txt pyproject.toml
	# Model-prerequisites fragment: keep spacy+nltk pinnable standalone (early,
	# cacheable model/nltk download in the image build).
	@printf 'nltk==%s\nspacy==%s\n' \
		"$$(grep -m1 '^nltk==' requirements-full.txt | cut -d= -f3)" \
		"$$(grep -m1 '^spacy==' requirements-full.txt | cut -d= -f3)" > requirements-models.txt

