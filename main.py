"""
Rate-limited public API wrapper (FastAPI)

Wraps the free Open-Meteo weather API (no upstream key needed) and adds:
  - API key authentication (X-API-Key header)
  - Per-key rate limiting (token bucket) with tiers
  - Standard rate limit headers + 429 with Retry-After
  - In-memory TTL cache so repeated lookups don't hit the upstream
  - Usage stats per key

Run:
    pip install fastapi uvicorn httpx
    ADMIN_TOKEN=change-me uvicorn main:app --reload

Try:
    curl -H "X-Admin-Token: change-me" -X POST "localhost:8000/admin/keys?tier=free"
    curl -H "X-API-Key: <key>" "localhost:8000/weather?lat=12.97&lon=77.59"
"""

import os
import secrets
import time
from dataclasses import dataclass, field

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "change-me")
UPSTREAM_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_TTL_SECONDS = 300  # 5 minutes

# tier -> (requests per minute, burst capacity)
TIERS = {
    "free": (10, 10),
    "pro": (60, 60),
}

app = FastAPI(title="Weather Wrapper API", version="1.0.0")


# --------------------------------------------------------------------------
# Token bucket rate limiter
# --------------------------------------------------------------------------

@dataclass
class Bucket:
    capacity: float
    refill_rate: float  # tokens per second
    tokens: float = 0.0
    last_refill: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        self.tokens = self.capacity

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    def consume(self, cost: float = 1.0) -> bool:
        self._refill()
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def seconds_until_available(self, cost: float = 1.0) -> float:
        self._refill()
        missing = max(0.0, cost - self.tokens)
        return missing / self.refill_rate

    def seconds_until_full(self) -> float:
        self._refill()
        return (self.capacity - self.tokens) / self.refill_rate


# --------------------------------------------------------------------------
# In-memory stores (swap for Redis/DB in production)
# --------------------------------------------------------------------------

@dataclass
class ApiKey:
    key: str
    tier: str
    created_at: float
    total_requests: int = 0
    rate_limited: int = 0


API_KEYS: dict[str, ApiKey] = {}
BUCKETS: dict[str, Bucket] = {}
CACHE: dict[str, tuple[float, dict]] = {}  # cache_key -> (expires_at, data)


def create_key(tier: str) -> ApiKey:
    key = "wk_" + secrets.token_urlsafe(24)
    API_KEYS[key] = ApiKey(key=key, tier=tier, created_at=time.time())
    per_minute, burst = TIERS[tier]
    BUCKETS[key] = Bucket(capacity=burst, refill_rate=per_minute / 60.0)
    return API_KEYS[key]


# A demo key so you can test immediately
DEMO = create_key("free")
print(f"Demo API key: {DEMO.key}")


# --------------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------------

def rate_limited_key(
    response: Response,
    x_api_key: str | None = Header(default=None),
) -> ApiKey:
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")

    api_key = API_KEYS[x_api_key]
    bucket = BUCKETS[x_api_key]
    per_minute, _ = TIERS[api_key.tier]
    api_key.total_requests += 1

    if not bucket.consume():
        api_key.rate_limited += 1
        retry_after = max(1, int(bucket.seconds_until_available() + 0.999))
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded",
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(per_minute),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(retry_after),
            },
        )

    response.headers["X-RateLimit-Limit"] = str(per_minute)
    response.headers["X-RateLimit-Remaining"] = str(int(bucket.tokens))
    response.headers["X-RateLimit-Reset"] = str(int(bucket.seconds_until_full() + 0.999))
    return api_key


def require_admin(x_admin_token: str | None = Header(default=None)) -> None:
    if not x_admin_token or not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(status_code=403, detail="Admin token required")


# --------------------------------------------------------------------------
# Upstream call with caching
# --------------------------------------------------------------------------

async def fetch_weather(lat: float, lon: float) -> tuple[dict, bool]:
    """Returns (data, from_cache)."""
    cache_key = f"{round(lat, 2)}:{round(lon, 2)}"
    now = time.time()

    hit = CACHE.get(cache_key)
    if hit and hit[0] > now:
        return hit[1], True

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        "timezone": "auto",
    }
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            r = await client.get(UPSTREAM_URL, params=params)
            r.raise_for_status()
    except httpx.HTTPError as exc:
        # Serve stale data if we have it, rather than failing outright
        if hit:
            return hit[1], True
        raise HTTPException(status_code=502, detail=f"Upstream error: {exc}")

    raw = r.json()
    current = raw.get("current", {})
    data = {
        "latitude": raw.get("latitude"),
        "longitude": raw.get("longitude"),
        "timezone": raw.get("timezone"),
        "time": current.get("time"),
        "temperature_c": current.get("temperature_2m"),
        "humidity_pct": current.get("relative_humidity_2m"),
        "wind_kmh": current.get("wind_speed_10m"),
        "weather_code": current.get("weather_code"),
    }
    CACHE[cache_key] = (now + CACHE_TTL_SECONDS, data)
    return data, False


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    """Unauthenticated, not rate limited."""
    return {"status": "ok"}


@app.get("/weather")
async def weather(
    response: Response,
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    api_key: ApiKey = Depends(rate_limited_key),
):
    data, cached = await fetch_weather(lat, lon)
    response.headers["X-Cache"] = "HIT" if cached else "MISS"
    return data


@app.get("/usage")
def usage(api_key: ApiKey = Depends(rate_limited_key)):
    per_minute, burst = TIERS[api_key.tier]
    return {
        "tier": api_key.tier,
        "limit_per_minute": per_minute,
        "burst": burst,
        "total_requests": api_key.total_requests,
        "rate_limited_requests": api_key.rate_limited,
    }


@app.post("/admin/keys", dependencies=[Depends(require_admin)])
def admin_create_key(tier: str = Query("free")):
    if tier not in TIERS:
        raise HTTPException(status_code=400, detail=f"tier must be one of {list(TIERS)}")
    k = create_key(tier)
    return {"api_key": k.key, "tier": k.tier}
