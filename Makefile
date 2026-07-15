.PHONY: format start_server start_client

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
		--host 0.0.0.0 \
		--port 8000 \
		--max-queue-size 32 \
		--retention-hours 168

start_client:
	python client.py \
		--server http://127.0.0.1:8000 \
		--image examples/03/image.jpg \
		--frame-num 361 \
		--seed 42 \
		--output output/result.mp4 \
		--prompt "A serene lakeside scene with a lone tree standing in calm water."
