# ML-7: Costing & Quoting with Live Data

Type: enhancement • Priority: High • Depends on: ML-6 • ETA: 4–6 days

## Objective
Replace placeholder pricing and irradiance with live sources; add geocoding and improve quote accuracy.

## Acceptance Criteria
- [ ] PVGIS v5.2 client with retry/backoff + cache; monthly/annual irradiance
- [ ] Cost catalog loader (CSV/DB) with VAT, installation overheads, margin, regional factor
- [ ] Geocoding from address/postcode → (lat, lon) and validation; caches results
- [ ] `/quote` wires these sources; provenance (versions) included in response
- [ ] Unit/integration tests using mocks/fixtures; README updated

## Tasks
1) PVGIS Service
   - HTTP client; cache with TTL; handle 429/5xx; integration test with recorded cassette
2) Cost Service
   - Load catalog from Supabase (preferred) or `data/costs_vYYYYMM.csv`
   - Compute totals: components + accessories + VAT + overheads + regional + margin
3) Geocoding
   - Nominatim or provider key; rate limit; cache results to Supabase `geocode_cache`
4) Quote Calculator
   - Include provenance fields (`cost_catalog_version`, `pvgis_timestamp`)
   - Update tests and README examples

## Risks
- External API limits; mitigate via caching and small test datasets
- Price variability; version the catalog and store in DB for reproducibility

Deliverables: Code, tests, docs. Labels: `ml-service`, `quoting`, `costing`