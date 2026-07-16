# Production Deployment

## Recommended Azure Topology

- Azure Container Apps for the FastAPI service
- Azure Container Registry for image storage
- Azure Key Vault for secrets
- Azure Cache for Redis for cache and session history
- MongoDB Atlas on Azure for vector and text search
- Azure Monitor / Log Analytics for logs and alerts

## Runtime Contract

Required environment variables:

- `MONGO_URI`
- `GROQ_API_KEY`

Optional but recommended:

- `REDIS_URL`
- `LANGCHAIN_API_KEY`
- `LANGCHAIN_PROJECT`
- `OPEN_ROUTER_API_KEY`
- `GOOGLE_API_KEY`
- `GEMINI_API`

Runtime settings:

- `PORT=8000`
- `ARISE_RELOAD=false`
- `ARISE_PRELOAD_MODELS=false`
- `LOG_AS_JSON=true`
- `LOG_TO_FILES=false`

## Local Container Validation

```bash
docker build -t arise-chatbot .
docker run --rm -p 8000:8000 --env-file .env arise-chatbot
```

Health endpoints:

- `GET /health`
- `GET /ready`

## Azure Deployment Sequence

1. Create Azure Container Registry.
2. Create Azure Container Apps environment.
3. Create Azure Key Vault and add app secrets.
4. Create Azure Cache for Redis.
5. Deploy MongoDB Atlas in the same region and confirm indexes.
6. Build and push the image.
7. Deploy the image to Azure Container Apps.
8. Validate `/health`, `/ready`, and a real query.

## Notes

- Session history now uses Redis when available and falls back to local memory only if Redis is unavailable.
- File logging is disabled by default for container deployments; logs go to stdout/stderr for Azure collection.
- Start with one worker and one replica, then scale based on memory and latency measurements.
