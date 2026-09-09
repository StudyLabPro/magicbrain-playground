SHELL := /bin/bash

# Порты, домен и режим публикации берутся из .env, чтобы `make status` проверял
# тот порт, который реально опубликован, а `make up` поднимал ровно тот набор
# файлов, который соответствует принятому решению о публикации. Файлов может
# ещё не быть — отсюда «-include».
-include .env
-include .env.build
export

MBP_PORT ?= 8007
MBP_API_PORT ?= 8008
STEPS ?= 100000

# Список --env-file собирается только из существующих файлов: иначе `make down`
# и `make logs` падали в свежем клоне и после `make clean`, то есть ровно
# тогда, когда они нужнее всего.
ENV_FILES := $(foreach f,.env .env.build,$(if $(wildcard $(f)),--env-file $(f)))
# Внешний маршрут — отдельный compose-файл, а не переменная в лейбле: пока
# публикация не включена, контейнер не состоит в сети обратного прокси.
# MBP_ROUTE_MODE=auth (по умолчанию) — маршрут под basic-auth;
# MBP_ROUTE_MODE=open — открытый показ без пароля, отдельное решение.
MBP_ROUTE_MODE ?= auth
ROUTE_FILE := $(if $(filter open,$(MBP_ROUTE_MODE)),docker-compose.public.yml,docker-compose.publish.yml)
COMPOSE_FILES := -f docker-compose.yml $(if $(filter true,$(MBP_TRAEFIK_ENABLE)),-f $(ROUTE_FILE))
COMPOSE := docker compose $(ENV_FILES) $(COMPOSE_FILES)

.PHONY: help corpora prepare state build compose-build up down logs status smoke runs-doc test clean clean-images

help:
	@echo "corpora      — собрать корпуса из законных источников (нужна сеть)"
	@echo "prepare      — снять снимок исходников MagicBrain в build/"
	@echo "build        — собрать образ (prepare выполняется автоматически)"
	@echo "up           — поднять стек уже собранным образом (пересборки нет)"
	@echo "down         — остановить"
	@echo "logs         — хвост логов"
	@echo "status       — состояние контейнеров и health"
	@echo "smoke        — прогнать все сценарии через HTTP (STEPS=$(STEPS))"
	@echo "runs-doc     — пересобрать docs/RUNS.md из фактических файлов прогонов"
	@echo "clean-images — удалить старые теги magicbrain-playground, кроме текущего"

corpora:
	python3 scripts/fetch_corpora.py

prepare:
	./scripts/prepare_build.sh
	@test -f .env || cp .env.example .env

# Каталоги состояния принадлежат пользователю контейнера (uid 10001). Раньше
# это был шаг раннбука «не забыть chown»; забытый шаг уводил контейнер в
# бесконечный рестарт.
state:
	@mkdir -p var/runs var/models
	@chown 10001:10001 var/runs var/models

# Сборка идёт через под-make: prepare может обновить .env.build, а текущий
# процесс make прочитал его при старте и уже экспортировал старый
# MBP_BUILD_ID — переменная окружения приоритетнее --env-file, и образ получил
# бы тег, которого нет в файле.
build: prepare
	@$(MAKE) --no-print-directory compose-build

compose-build:
	$(COMPOSE) build
	@echo "образ magicbrain-playground:$(MBP_BUILD_ID) собран"

# up НЕ зависит от prepare: prepare переписывал бы тег образа, и разворачивалось
# бы не то, что собирали и проверяли. Для запуска достаточно .env.build и уже
# собранного образа — репозиторий MagicBrain при этом не нужен.
up: state
	@test -f .env || cp .env.example .env
	@test -f .env.build || { echo "нет .env.build: сначала 'make build'"; exit 1; }
	@docker image inspect magicbrain-playground:$(MBP_BUILD_ID) >/dev/null 2>&1 || { \
		echo "нет образа magicbrain-playground:$(MBP_BUILD_ID): сначала 'make build'"; exit 1; }
	$(COMPOSE) up -d
	@echo "плейграунд: http://127.0.0.1:$(MBP_PORT)/  (маршрут наружу: $(if $(filter true,$(MBP_TRAEFIK_ENABLE)),ВКЛЮЧЁН на $(MBP_DOMAIN),выключен))"

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f --tail=100

status:
	$(COMPOSE) ps
	@curl -fsS http://127.0.0.1:$(MBP_PORT)/health && echo
	@curl -fsS http://127.0.0.1:$(MBP_API_PORT)/health && echo

smoke:
	MBP_BASE=http://127.0.0.1:$(MBP_PORT) python3 scripts/run_scenarios.py --steps $(STEPS)

runs-doc:
	python3 scripts/render_runs.py

test:
	python3 -m pytest -q tests/

# .env.build не удаляем: без него нельзя остановить уже поднятый стек.
clean:
	rm -rf build

clean-images:
	@docker images --format '{{.Repository}}:{{.Tag}}' magicbrain-playground \
		| grep -v ':$(MBP_BUILD_ID)$$' \
		| xargs -r docker image rm
	@echo "висячие слои от прошлых сборок остаются: 'docker image prune -f' — решение того, кто отвечает за хост"
