"""
ADVE Production Auth + Billing
================================
Simple but real: API keys, usage tracking, Razorpay billing.
This is what turns ADVE from a demo into a product.

Tables:
  users         — registered users
  api_keys      — API keys (one per user tier)
  usage         — per-request usage logging
  subscriptions — active subscriptions

Add to server.py imports and use the middleware.
"""

import os
import uuid
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import sqlite3


# ── Database setup ────────────────────────────────────────────────────────────

def init_auth_db(db_path: str = "data/auth.db"):
    db = sqlite3.connect(db_path, check_same_thread=False)

    db.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id           TEXT PRIMARY KEY,
        email        TEXT UNIQUE NOT NULL,
        name         TEXT,
        created_at   REAL,
        tier         TEXT DEFAULT 'free'
    );

    CREATE TABLE IF NOT EXISTS api_keys (
        key_hash     TEXT PRIMARY KEY,
        user_id      TEXT,
        label        TEXT,
        created_at   REAL,
        is_active    INTEGER DEFAULT 1,
        FOREIGN KEY (user_id) REFERENCES users(id)
    );

    CREATE TABLE IF NOT EXISTS usage (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id      TEXT,
        endpoint     TEXT,
        video_mins   REAL DEFAULT 0,
        timestamp    REAL
    );

    CREATE TABLE IF NOT EXISTS subscriptions (
        user_id      TEXT PRIMARY KEY,
        tier         TEXT,
        started_at   REAL,
        expires_at   REAL,
        razorpay_id  TEXT
    );
    """)
    db.commit()
    return db


# ── Tier limits ───────────────────────────────────────────────────────────────

TIER_LIMITS = {
    "free": {
        "videos_per_month": 5,
        "minutes_per_month": 30,
        "searches_per_day":  50,
        "price_inr":         0,
    },
    "starter": {
        "videos_per_month": 50,
        "minutes_per_month": 300,
        "searches_per_day":  500,
        "price_inr":         2000,
    },
    "growth": {
        "videos_per_month": 500,
        "minutes_per_month": 3000,
        "searches_per_day":  5000,
        "price_inr":         10000,
    },
    "business": {
        "videos_per_month": -1,    # unlimited
        "minutes_per_month": -1,
        "searches_per_day":  -1,
        "price_inr":         40000,
    },
}


# ── Auth manager ──────────────────────────────────────────────────────────────

class AuthManager:
    def __init__(self, db_path: str = "data/auth.db"):
        self.db = init_auth_db(db_path)

    def create_user(self, email: str, name: str = "") -> dict:
        user_id = str(uuid.uuid4())
        try:
            self.db.execute(
                "INSERT INTO users VALUES (?,?,?,?,?)",
                (user_id, email, name, time.time(), "free")
            )
            self.db.commit()
        except sqlite3.IntegrityError:
            raise HTTPException(400, f"Email already registered: {email}")

        api_key = self.create_api_key(user_id, "default")
        return {"user_id": user_id, "api_key": api_key}

    def create_api_key(self, user_id: str, label: str = "default") -> str:
        raw_key  = f"adve-{uuid.uuid4().hex}"
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

        self.db.execute(
            "INSERT INTO api_keys VALUES (?,?,?,?,?)",
            (key_hash, user_id, label, time.time(), 1)
        )
        self.db.commit()
        return raw_key  # return raw key ONCE — never stored in plaintext

    def verify_key(self, raw_key: str) -> Optional[dict]:
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        row = self.db.execute(
            """SELECT u.id, u.email, u.tier
               FROM api_keys k JOIN users u ON k.user_id = u.id
               WHERE k.key_hash=? AND k.is_active=1""",
            (key_hash,)
        ).fetchone()

        if not row:
            return None

        return {"user_id": row[0], "email": row[1], "tier": row[2]}

    def check_limits(self, user_id: str, tier: str, endpoint: str) -> bool:
        limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])

        if endpoint == "search":
            limit = limits["searches_per_day"]
            if limit == -1:
                return True
            # Count today's searches
            today_start = time.time() - 86400
            count = self.db.execute(
                "SELECT COUNT(*) FROM usage WHERE user_id=? AND endpoint='search' AND timestamp>?",
                (user_id, today_start)
            ).fetchone()[0]
            return count < limit

        if endpoint == "index":
            limit = limits["minutes_per_month"]
            if limit == -1:
                return True
            month_start = time.time() - 30 * 86400
            total_mins = self.db.execute(
                "SELECT SUM(video_mins) FROM usage WHERE user_id=? AND endpoint='index' AND timestamp>?",
                (user_id, month_start)
            ).fetchone()[0] or 0
            return total_mins < limit

        return True

    def log_usage(self, user_id: str, endpoint: str, video_mins: float = 0):
        self.db.execute(
            "INSERT INTO usage VALUES (NULL,?,?,?,?)",
            (user_id, endpoint, video_mins, time.time())
        )
        self.db.commit()

    def get_usage_stats(self, user_id: str) -> dict:
        tier = self.db.execute(
            "SELECT tier FROM users WHERE id=?", (user_id,)
        ).fetchone()[0]

        limits = TIER_LIMITS.get(tier, TIER_LIMITS["free"])

        month_start  = time.time() - 30 * 86400
        today_start  = time.time() - 86400

        mins_used = self.db.execute(
            "SELECT SUM(video_mins) FROM usage WHERE user_id=? AND endpoint='index' AND timestamp>?",
            (user_id, month_start)
        ).fetchone()[0] or 0

        searches_today = self.db.execute(
            "SELECT COUNT(*) FROM usage WHERE user_id=? AND endpoint='search' AND timestamp>?",
            (user_id, today_start)
        ).fetchone()[0]

        return {
            "tier":             tier,
            "price_inr":        limits["price_inr"],
            "minutes_used":     round(float(mins_used), 1),
            "minutes_limit":    limits["minutes_per_month"],
            "searches_today":   searches_today,
            "searches_limit":   limits["searches_per_day"],
        }


# ── FastAPI dependency ────────────────────────────────────────────────────────

security     = HTTPBearer()
auth_manager = AuthManager()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> dict:
    """FastAPI dependency — use in any endpoint that requires auth."""
    user = auth_manager.verify_key(credentials.credentials)
    if not user:
        raise HTTPException(401, "Invalid or expired API key")
    return user


def require_limit(endpoint: str):
    """Dependency factory for rate limiting."""
    def checker(user: dict = Depends(get_current_user)) -> dict:
        if not auth_manager.check_limits(user["user_id"], user["tier"], endpoint):
            raise HTTPException(
                429,
                f"Usage limit reached for {endpoint}. "
                f"Upgrade at adve.in/pricing"
            )
        return user
    return checker


# ── Razorpay payment integration ─────────────────────────────────────────────

def create_razorpay_order(user_id: str, tier: str) -> dict:
    """
    Create a Razorpay payment order.
    Returns: {order_id, amount, currency, key_id}
    """
    try:
        import razorpay
        client = razorpay.Client(auth=(
            os.getenv("RAZORPAY_KEY_ID"),
            os.getenv("RAZORPAY_KEY_SECRET"),
        ))

        amount_paise = TIER_LIMITS[tier]["price_inr"] * 100  # Razorpay uses paise

        order = client.order.create({
            "amount":   amount_paise,
            "currency": "INR",
            "notes":    {"user_id": user_id, "tier": tier},
        })

        return {
            "order_id": order["id"],
            "amount":   amount_paise,
            "currency": "INR",
            "key_id":   os.getenv("RAZORPAY_KEY_ID"),
        }

    except ImportError:
        raise HTTPException(500, "Razorpay not installed. pip install razorpay")


def verify_razorpay_payment(
    order_id:   str,
    payment_id: str,
    signature:  str,
    user_id:    str,
    tier:       str,
) -> bool:
    """Verify payment signature and activate subscription."""
    try:
        import razorpay, hmac, hashlib as hs

        key_secret = os.getenv("RAZORPAY_KEY_SECRET", "")
        msg        = f"{order_id}|{payment_id}"
        generated  = hmac.new(
            key_secret.encode(), msg.encode(), hs.sha256
        ).hexdigest()

        if generated != signature:
            return False

        # Activate subscription
        db = auth_manager.db
        expires_at = time.time() + 30 * 86400  # 30 days

        db.execute(
            "INSERT OR REPLACE INTO subscriptions VALUES (?,?,?,?,?)",
            (user_id, tier, time.time(), expires_at, payment_id)
        )
        db.execute(
            "UPDATE users SET tier=? WHERE id=?",
            (tier, user_id)
        )
        db.commit()

        return True

    except Exception:
        return False


# ── Add these endpoints to server.py ─────────────────────────────────────────

AUTH_ENDPOINTS = """
from auth import AuthManager, get_current_user, require_limit
from auth import create_razorpay_order, verify_razorpay_payment
from pydantic import BaseModel

class RegisterRequest(BaseModel):
    email: str
    name:  str = ""

class PaymentVerifyRequest(BaseModel):
    order_id:   str
    payment_id: str
    signature:  str
    tier:       str


@app.post("/v1/register")
async def register(request: RegisterRequest):
    result = auth_manager.create_user(request.email, request.name)
    return {
        "message":  "Account created",
        "api_key":  result["api_key"],
        "warning":  "Save this API key — it won't be shown again",
    }


@app.get("/v1/usage")
async def get_usage(user: dict = Depends(get_current_user)):
    return auth_manager.get_usage_stats(user["user_id"])


@app.post("/v1/billing/create-order/{tier}")
async def create_order(
    tier: str,
    user: dict = Depends(get_current_user),
):
    if tier not in ["starter", "growth", "business"]:
        raise HTTPException(400, "Invalid tier")
    return create_razorpay_order(user["user_id"], tier)


@app.post("/v1/billing/verify-payment")
async def verify_payment(
    request: PaymentVerifyRequest,
    user:    dict = Depends(get_current_user),
):
    ok = verify_razorpay_payment(
        request.order_id, request.payment_id,
        request.signature, user["user_id"], request.tier
    )
    if not ok:
        raise HTTPException(400, "Payment verification failed")
    return {"message": f"Subscription activated: {request.tier}"}


# Update existing endpoints to require auth:
@app.post("/v1/index/video")
async def index_video_authed(
    file: UploadFile = File(...),
    user: dict = Depends(require_limit("index")),
):
    # ... existing indexing logic ...
    # Add: auth_manager.log_usage(user["user_id"], "index", video_duration_mins)
    pass


@app.post("/v1/search/text")
async def search_text_authed(
    request: TextSearchRequest,
    user:    dict = Depends(require_limit("search")),
):
    # ... existing search logic ...
    # Add: auth_manager.log_usage(user["user_id"], "search")
    pass
"""
