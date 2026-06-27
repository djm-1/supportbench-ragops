# SupportBench on the QuotePilot DigitalOcean Droplet

This deployment reuses the existing QuotePilot droplet services:

- Caddy from `~/QuotePilot/deploy/digitalocean`
- the existing Postgres container named `digitalocean-db-1`
- the existing Docker network `digitalocean_default`

SupportBench adds only one backend container. The Next.js frontend should run on Vercel.

## 1. Create the backend DNS name

Create another DuckDNS subdomain and point it at the same droplet IP:

```text
supportbench.duckdns.org -> 192.81.217.233
```

Use your actual droplet IP if it changes.

## 2. Create a separate database in the shared Postgres container

SSH into the droplet, then open `psql` from the QuotePilot deployment folder:

```bash
cd ~/QuotePilot/deploy/digitalocean
docker compose exec db psql -U quotepilot -d postgres
```

Run this SQL. Replace the password with the same strong password you will put in SupportBench's `.env`.

```sql
CREATE USER supportbench WITH PASSWORD 'replace-with-a-long-random-password';
CREATE DATABASE supportbench OWNER supportbench;
GRANT ALL PRIVILEGES ON DATABASE supportbench TO supportbench;
\q
```

If you already created the user earlier, use this instead:

```sql
ALTER USER supportbench WITH PASSWORD 'replace-with-a-long-random-password';
CREATE DATABASE supportbench OWNER supportbench;
GRANT ALL PRIVILEGES ON DATABASE supportbench TO supportbench;
\q
```

## 3. Configure SupportBench backend env

Clone the SupportBench repo on the droplet:

```bash
cd ~
git clone https://github.com/djm-1/supportbench-ragops.git SupportBench
cd ~/SupportBench/deploy/digitalocean
cp .env.example .env
nano .env
```

Minimum production env:

```env
SUPPORTBENCH_DOMAIN=supportbench.duckdns.org
SUPPORTBENCH_POSTGRES_DB=supportbench
SUPPORTBENCH_POSTGRES_USER=supportbench
SUPPORTBENCH_POSTGRES_PASSWORD=replace-with-a-long-random-password
CORS_ALLOWED_ORIGINS=https://your-supportbench-frontend.vercel.app
USE_REAL_MODELS=false
USE_PINECONE=false
EVAL_JUDGE_PROVIDER=deterministic
```

Leave provider API keys blank for the first deployment. This keeps the public demo stable and cost-free.

## 4. Start SupportBench backend

```bash
cd ~/SupportBench/deploy/digitalocean
docker compose up -d --build
docker compose ps
```

Validate the backend container directly from Caddy's shared Docker network:

```bash
docker run --rm --network digitalocean_default curlimages/curl:8.10.1 http://supportbench-api:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## 5. Add the Caddy route

Edit QuotePilot's Caddyfile:

```bash
nano ~/QuotePilot/deploy/digitalocean/Caddyfile
```

Add this block:

```caddy
supportbench.duckdns.org {
	encode zstd gzip
	reverse_proxy supportbench-api:8000
}
```

Then reload Caddy:

```bash
cd ~/QuotePilot/deploy/digitalocean
docker compose up -d caddy
```

Public smoke test:

```bash
curl https://supportbench.duckdns.org/health
```

Expected response:

```json
{"status":"ok"}
```

## 6. Deploy the frontend on Vercel

Import the GitHub repo into Vercel with these settings:

```text
Root Directory: frontend
Framework Preset: Next.js
Build Command: npm run build
Output Directory: .next
```

Set this Vercel environment variable:

```env
NEXT_PUBLIC_API_BASE_URL=https://supportbench.duckdns.org
```

After Vercel gives you the final frontend URL, update SupportBench backend CORS:

```bash
cd ~/SupportBench/deploy/digitalocean
nano .env
```

Set:

```env
CORS_ALLOWED_ORIGINS=https://your-supportbench-frontend.vercel.app
```

Restart the backend:

```bash
docker compose up -d
```

## 7. First live test

1. Open the Vercel frontend.
2. Confirm the backend status loads.
3. Run ingestion from the app.
4. Run a 1-3 question benchmark first.
5. Check Decision, Test Cases, Trace, and CSV export.

## Memory Notes

On the 1 GB droplet, keep SupportBench in deterministic/local mode unless you upgrade:

```env
USE_REAL_MODELS=false
USE_PINECONE=false
EVAL_JUDGE_PROVIDER=deterministic
```

Check memory after starting both apps:

```bash
free -h
docker stats --no-stream
```

If available memory stays above roughly `200Mi` and swap stays low, the setup is acceptable for portfolio traffic. If swap grows continuously or requests become slow, upgrade to a 2 GB droplet.
