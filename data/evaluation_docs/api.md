# API operations — synthetic evaluation fixture
The Standard API rate limit is 120 requests per minute per token.
Clients receiving HTTP 429 should honor the Retry-After header before retrying.
Idempotency keys expire after 24 hours. Repeated requests with the same idempotency key reuse the original result.
API tokens can be revoked by workspace administrators. Pagination uses an opaque cursor rather than an offset.
