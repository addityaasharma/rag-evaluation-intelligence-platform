# Northwind Cloud API Rate Limits

## Limits per plan
- Starter: 60 requests per minute
- Pro: 600 requests per minute
- Enterprise: 6,000 requests per minute

Each plan allows bursts of up to 2x the limit for 10 seconds.

## Exceeding the limit
Requests over the limit fail with HTTP 429 and a `Retry-After` header giving the number of seconds to wait before retrying.

## Authentication and pagination
The API authenticates with bearer API keys. Keys expire after 90 days by default and should be rotated before then. List endpoints return at most 200 items per page.
