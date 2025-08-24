# ML-8: Security, Ops, and Persistence (Supabase)

Type: enhancement • Priority: High • Depends on: ML-6, ML-7 • ETA: 4–7 days

## Objective
Secure the API, add persistence for quotes and measurements, and improve observability with Supabase as the data store.

## Acceptance Criteria
- [ ] Supabase schema created/migrated as per `SUPABASE_SPEC.md` (tables + RLS)
- [ ] Save quotes and items; `GET /quote/{id}` returns stored record
- [ ] Optional PDF export for quotes (basic template)
- [ ] API key auth (header `X-API-Key`) with hashed keys in DB; strict CORS allow list
- [ ] Request logging with request_id persisted; basic metrics counters
- [ ] Uptime checks configured for prod `/health`; README updated with ops steps

## Tasks
1) Persistence
   - Add Supabase client (service role key) on server-side only
   - On `/quote`, write `properties`, `roof_measurements`, `quotes`, `quote_items`
   - New endpoint: `GET /quote/{id}` (read from DB)
2) Security
   - CORS `CORS_ALLOW_ORIGINS` env
   - API key middleware (lookup hashed keys in `api_keys`)
3) Observability
   - Structured JSON logs; write request summary to `request_logs`
   - Optional Prometheus middleware
4) Migrations & RLS
   - Migration SQL for schema; enable RLS; policy for service role only
   - Sample seed data for catalogs

## Risks
- Leaking service key; mitigate via environment secrets and server-only usage
- Latency on DB writes; use simple sync path first, consider queue later

Deliverables: Code, migrations, tests, docs. Labels: `security`, `ops`, `persistence`, `supabase`