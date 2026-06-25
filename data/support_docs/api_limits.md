# API Limits

## Rate Limits

Free workspaces are limited to 60 API requests per minute. Pro workspaces are limited to 600 API requests per minute. Enterprise workspaces receive a custom limit defined in the order form.

## Burst Behavior

Short bursts above the rate limit may be accepted, but sustained traffic above the limit returns HTTP 429 responses. Clients should use exponential backoff and retry after the time specified in the Retry-After header.

## API Tokens

API tokens inherit the permissions of the user who created them. Tokens can be revoked from Settings > Developers > API Tokens. Token values are shown only once when created.

## Webhooks

Webhook delivery is retried up to five times with exponential backoff. A webhook endpoint is disabled automatically after 20 consecutive delivery failures.
