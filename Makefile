.PHONY: start stop restart redeploy build status logs shell

# Start the service in the background
start:
	docker compose up -d

# Stop the service
stop:
	docker compose stop

# Restart the container (no rebuild)
restart:
	docker compose restart

# Full redeploy: stop, rebuild, and start
redeploy:
	docker compose stop
	docker compose build
	docker compose up -d

# Build the image without starting
build:
	docker compose build

# Check service and model status
status:
	@docker compose ps
	@echo ""
	@curl -s http://localhost:8335/api/status | python3 -m json.tool 2>/dev/null || echo "Service not reachable"

# Tail container logs
logs:
	docker compose logs -f

# Open a shell in the running container
shell:
	docker compose exec qwen3tts bash
