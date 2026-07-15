.PHONY: format start_server start_client health list job download

SERVER_URL ?= http://127.0.0.1:8000
LIMIT ?= 20
JOB_ID ?=
OUTPUT ?= /mnt/outputs/lingbot-world-v2/downloads/$(JOB_ID).mp4

format:
	isort generate.py wan
	yapf -i -r *.py generate.py wan

start_server:
	CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
	FSDP_SHARDING_STRATEGY=FULL_SHARD \
	NCCL_DEBUG=WARN \
	torchrun --nproc_per_node=8 server.py \
		--ckpt-dir lingbot-world-v2-14b-causal-fast \
		--action-path examples/03 \
		--data-dir /mnt/workspace/lingbot-world-service \
		--output-dir /mnt/outputs/lingbot-world-v2 \
		--host 0.0.0.0 \
		--port 8000 \
		--max-queue-size 32 \
		--retention-hours 168

start_client:
	python client.py \
		--server $(SERVER_URL) \
		--image examples/03/image.jpg \
		--frame-num 361 \
		--seed 42 \
		--output-dir /mnt/outputs/lingbot-world-v2/downloads \
		--prompt "A serene lakeside scene with a lone tree standing in calm water."

health:
	@curl --fail --silent --show-error --write-out '\n' \
		"$(SERVER_URL)/health"

list:
	@curl --fail --silent --show-error --write-out '\n' \
		"$(SERVER_URL)/v1/jobs?limit=$(LIMIT)"

job:
	@test -n "$(JOB_ID)" || \
		(echo "JOB_ID is required; example: make job JOB_ID=<job-id>" >&2; exit 2)
	@curl --fail --silent --show-error --write-out '\n' \
		"$(SERVER_URL)/v1/jobs/$(JOB_ID)"

download:
	@test -n "$(JOB_ID)" || \
		(echo "JOB_ID is required; example: make download JOB_ID=<job-id>" >&2; exit 2)
	@mkdir -p "$(dir $(OUTPUT))"
	@curl --fail --silent --show-error \
		--output "$(OUTPUT).part" \
		"$(SERVER_URL)/v1/jobs/$(JOB_ID)/video"
	@mv "$(OUTPUT).part" "$(OUTPUT)"
	@echo "video saved to $(OUTPUT)"
