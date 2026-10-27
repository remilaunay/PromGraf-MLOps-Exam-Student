.DEFAULT_GOAL := all

COMPOSE ?= docker compose
PERIOD ?= week1_february
TRAFFIC_COUNT ?= 50
API_URL ?= http://localhost:8080

.PHONY: all stop evaluation traffic fire-alert status logs

all:
	$(COMPOSE) up --build --detach

stop:
	$(COMPOSE) down

evaluation:
	$(COMPOSE) --profile tools run --build --rm evaluation python run_evaluation.py --period $(PERIOD)

traffic:
	$(COMPOSE) --profile tools run --build --rm evaluation python run_evaluation.py --period $(PERIOD) --traffic-count $(TRAFFIC_COUNT)

fire-alert:
	curl --fail --silent --show-error --write-out '\n' -X POST "$(API_URL)/trigger-drift"

status:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs --follow
