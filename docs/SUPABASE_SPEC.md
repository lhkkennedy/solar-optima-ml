# Supabase Schema Spec (SolarOptima ML)

Goal: persist quotes, cache external data, store catalog pricing, and provide minimal auth/ops.

## Existing tables (from screenshot)
- batteries
- inverters
- solar_panels
- tariff_rates
- property_analysis (assumed partial)
- system_quotes (assumed partial)

Retain these and add/normalize as below.

## New/updated tables

### properties
- id UUID PK default gen_random_uuid()
- address text not null
- postcode text not null
- latitude double precision
- longitude double precision
- property_type text
- occupancy text
- created_at timestamptz default now()
- updated_at timestamptz default now()

Index: (postcode), (latitude, longitude)

### quotes
- id UUID PK default gen_random_uuid()
- property_id UUID FK → properties(id) on delete cascade
- quote_number text unique not null (e.g., SOL-YYYY-XXXXXX)
- system_kwp numeric(6,2) not null
- panel_count integer not null
- inverter_sku text not null
- battery_sku text null
- estimated_yearly_kwh numeric(10,2) not null
- solar_fraction numeric(4,3) not null
- total_cost_gbp numeric(12,2) not null
- installation_cost_gbp numeric(12,2) not null
- battery_cost_gbp numeric(12,2)
- annual_savings_gbp numeric(12,2) not null
- payback_years numeric(6,2) not null
- roi_percent numeric(6,2) not null
- data jsonb not null  -- full API response snapshot
- valid_until date
- created_at timestamptz default now()
- updated_at timestamptz default now()

Index: (property_id), (created_at desc)

### quote_items
- id UUID PK default gen_random_uuid()
- quote_id UUID FK → quotes(id) on delete cascade
- sku text not null
- description text not null
- quantity integer not null
- unit_cost_gbp numeric(12,2) not null
- total_cost_gbp numeric(12,2) not null

Index: (quote_id)

### roof_measurements
- id UUID PK default gen_random_uuid()
- property_id UUID FK → properties(id) on delete cascade
- pitch_degrees numeric(5,2) not null
- area_m2 numeric(10,2) not null
- roof_type text not null
- orientation text not null
- height_m numeric(10,2) null
- slope_percent numeric(6,2) null
- segmentation_confidence numeric(4,3) null
- mask_png_base64 text null  -- optional storage of mask
- created_at timestamptz default now()

Index: (property_id, created_at desc)

### irradiance_cache
- id UUID PK default gen_random_uuid()
- latitude double precision not null
- longitude double precision not null
- monthly jsonb not null
- annual_kwh_m2 numeric(10,2) not null
- optimal_angle numeric(5,2) not null
- ttl_expires_at timestamptz not null
- created_at timestamptz default now()

Index: (latitude, longitude), (ttl_expires_at)

### geocode_cache
- id UUID PK default gen_random_uuid()
- address text not null
- postcode text not null
- latitude double precision not null
- longitude double precision not null
- provider text not null
- ttl_expires_at timestamptz not null
- created_at timestamptz default now()

Index: (postcode), (address)

### api_keys
- id UUID PK default gen_random_uuid()
- label text not null
- hashed_key text not null  -- store hash only
- role text not null default 'server'
- created_at timestamptz default now()
- revoked_at timestamptz

Unique: (hashed_key)

### request_logs (minimal)
- id bigserial PK
- request_id text
- path text not null
- status_code integer not null
- latency_ms integer
- created_at timestamptz default now()

Index: (created_at desc), (path)

## Catalog tables (normalize optional)
- batteries(id, sku unique, description, capacity_kwh, unit_cost_gbp, install_cost_gbp, warranty_years, supplier, updated_at)
- inverters(id, sku unique, description, kw_rating, unit_cost_gbp, install_cost_gbp, warranty_years, supplier, updated_at)
- solar_panels(id, sku unique, description, watt, area_m2, unit_cost_gbp, warranty_years, supplier, updated_at)
- tariff_rates(id, region, import_p_per_kwh, export_p_per_kwh, valid_from, valid_to)

## Row Level Security (RLS)
- If public API (no users), keep tables restricted and only allow service role (via `supabase_service_role` key) from server-side
- Enable RLS on all tables; create policies that allow only service role to read/write

## Example DDL snippets
```sql
create table if not exists properties (
  id uuid primary key default gen_random_uuid(),
  address text not null,
  postcode text not null,
  latitude double precision,
  longitude double precision,
  property_type text,
  occupancy text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);
create index if not exists idx_properties_postcode on properties(postcode);

create table if not exists quotes (
  id uuid primary key default gen_random_uuid(),
  property_id uuid references properties(id) on delete cascade,
  quote_number text unique not null,
  system_kwp numeric(6,2) not null,
  panel_count integer not null,
  inverter_sku text not null,
  battery_sku text,
  estimated_yearly_kwh numeric(10,2) not null,
  solar_fraction numeric(4,3) not null,
  total_cost_gbp numeric(12,2) not null,
  installation_cost_gbp numeric(12,2) not null,
  battery_cost_gbp numeric(12,2),
  annual_savings_gbp numeric(12,2) not null,
  payback_years numeric(6,2) not null,
  roi_percent numeric(6,2) not null,
  data jsonb not null,
  valid_until date,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);
create index if not exists idx_quotes_property on quotes(property_id);
```

## API usage plan
- ML-8 will add endpoints:
  - POST /quote → save to `properties`, `roof_measurements`, `quotes`, `quote_items`
  - GET /quote/{id} → read from `quotes` and `quote_items`
  - GET /property/{id}/latest-measurement → `roof_measurements` latest
- Server uses Supabase service role key only (no client exposure).