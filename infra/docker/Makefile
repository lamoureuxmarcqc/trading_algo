.PHONY: up down logs ps pull-model health

up:
	docker compose -f docker-compose.yml up -d

full:
	docker compose -f docker-compose.yml --profile full up -d

down:
	docker compose -f docker-compose.yml down

logs:
	docker compose -f docker-compose.yml logs -f

ps:
	docker compose -f docker-compose.yml ps

pull-model:
	docker exec lamoureux-ollama ollama pull $${OLLAMA_MODEL:-llama3.1:8b}

health:
	@echo "→ AI Gateway"; curl -s http://localhost:8001/health | head -c 400; echo
	@echo "→ Qdrant";    curl -s http://localhost:6333/healthz; echo
	@echo "→ Postgres";  docker exec lamoureux-postgres pg_isready -U lamoureux

smoke:
	curl -s -X POST http://localhost:8001/v1/inference \
	  -H "Content-Type: application/json" \
	  -d '{"prompt":"Say hello in one word.","classification":"PUBLIC","tenant_id":"smoke"}' | head -c 400
