import logging
from datetime import datetime, timedelta, timezone, UTC
from sqlalchemy.orm import Session
from database import SessionLocal
from models import ApiUsage
import os

logger = logging.getLogger(__name__)

class QuotaService:
    def __init__(self):
        # Strict user limits: max 20/day per provider
        # yfinance limits: ~2000 requests/hour per IP, but we throttle to be safe
        # We use higher limits now with aggressive caching to prevent rate limits
        # Based on research: Yahoo allows ~2000/hour, 10000-50000/day is safe
        self.defaults = {
            "google": {"rpm": 15, "tpm": 1000000, "rpd": 20},
            "groq": {"rpm": 30, "tpm": 144000, "rpd": 20},
            "sambanova": {"rpm": 15, "tpm": 50000, "rpd": 20},
            "yfinance": {"rpm": 30, "tpm": 500, "rpd": 10000},  # 30 req/min (2/sec), 10000/day (safe per research)
        }

    def _get_usage(self, db: Session, model_name: str) -> ApiUsage:
        usage = db.query(ApiUsage).filter(ApiUsage.model_name == model_name).first()
        if not usage:
            limits = self.defaults.get(model_name, {"rpm": 5, "tpm": 250000, "rpd": 20})
            usage = ApiUsage(
                model_name=model_name,
                limit_rpm=limits["rpm"],
                limit_tpm=limits["tpm"],
                limit_rpd=limits["rpd"],
                minute_requests=0,  # Explicitly initialize to 0
                minute_tokens=0,    # Explicitly initialize to 0
                day_requests=0,    # Explicitly initialize to 0
                last_request_at=None  # Will be set on first request
            )
            db.add(usage)
            db.commit()
            db.refresh(usage)
            logger.debug(f"Created new ApiUsage record for {model_name}")
        return usage

    def _get_now(self) -> datetime:
        """Get current time as naive datetime for consistent comparison."""
        return datetime.now(UTC).replace(tzinfo=None)

    def _reset_if_needed(self, db: Session, usage: ApiUsage) -> bool:
        """Reset counters if needed based on time. Returns True if reset occurred."""
        now = self._get_now()
        reset_occurred = False
        
        # Reset minute counters if a minute has passed or if last_request_at is None
        if usage.last_request_at is None:
            logger.debug(f"Resetting minute counters for {usage.model_name}: last_request_at is None")
            usage.minute_requests = 0
            usage.minute_tokens = 0
            reset_occurred = True
        else:
            # Ensure last_request_at is naive for comparison
            last_req = usage.last_request_at
            if last_req.tzinfo is not None:
                last_req = last_req.replace(tzinfo=None)
            
            time_diff = now - last_req
            if time_diff > timedelta(minutes=1):
                logger.debug(f"Resetting minute counters for {usage.model_name}: {time_diff} > 1 minute")
                usage.minute_requests = 0
                usage.minute_tokens = 0
                reset_occurred = True
        
        # Reset day counters if it's a new calendar day
        if usage.last_request_at is None:
            logger.debug(f"Resetting day counters for {usage.model_name}: last_request_at is None")
            usage.day_requests = 0
            reset_occurred = True
        else:
            # Ensure last_request_at is naive for comparison
            last_req = usage.last_request_at
            if last_req.tzinfo is not None:
                last_req = last_req.replace(tzinfo=None)
            
            if now.date() > last_req.date():
                logger.debug(f"Resetting day counters for {usage.model_name}: new day detected")
                usage.day_requests = 0
                reset_occurred = True
        
        return reset_occurred

    def check_quota(self, model_name: str) -> dict:
        """Check if we have enough quota for 1 request."""
        db = SessionLocal()
        try:
            usage = self._get_usage(db, model_name)
            now = self._get_now()

            # Reset counters if needed
            self._reset_if_needed(db, usage)
            
            # Commit the reset if any
            db.commit()

            # Debug logging to see current quota values
            logger.debug(f"Quota check for {model_name}: minute_requests={usage.minute_requests}/{usage.limit_rpm}, "
                        f"day_requests={usage.day_requests}/{usage.limit_rpd}, last_request_at={usage.last_request_at}")

            can_call = (
                usage.minute_requests < usage.limit_rpm and
                usage.day_requests < usage.limit_rpd
            )

            status = {
                "can_call": can_call,
                "model": model_name,  # Using model_name here actually means provider now
                "remaining_rpm": max(0, usage.limit_rpm - usage.minute_requests),
                "remaining_rpd": max(0, usage.limit_rpd - usage.day_requests),
                "remaining_tpm": max(0, usage.limit_tpm - usage.minute_tokens),
                "limit_rpm": usage.limit_rpm,
                "limit_rpd": usage.limit_rpd,
                "limit_tpm": usage.limit_tpm,
                "used_rpm_pct": round((usage.minute_requests / usage.limit_rpm) * 100, 1) if usage.limit_rpm > 0 else 0,
                "used_rpd_pct": round((usage.day_requests / usage.limit_rpd) * 100, 1) if usage.limit_rpd > 0 else 0,
                "low_quota": max(0, usage.limit_rpd - usage.day_requests) < 3
            }
            
            # Refresh from DB to get any updated values after commit
            db.refresh(usage)
            logger.debug(f"Quota status for {model_name}: can_call={can_call}, remaining_rpm={status['remaining_rpm']}, remaining_rpd={status['remaining_rpd']}")
            
            return status
        finally:
            db.close()

    def log_usage(self, model_name: str, tokens: int = 0):
        """Update usage after a successful call."""
        db = SessionLocal()
        try:
            usage = self._get_usage(db, model_name)
            now = self._get_now()

            # Reset counters if needed before incrementing
            self._reset_if_needed(db, usage)

            usage.minute_requests += 1
            usage.day_requests += 1
            usage.minute_tokens += tokens
            usage.last_request_at = now
            
            db.commit()
            logger.info(f"Logged quota for {model_name}: +1 req, +{tokens} tokens. Day total: {usage.day_requests}")
        finally:
            db.close()

    def reset_quota(self, model_name: str = None):
        """Manually reset quota counters. If model_name is None, reset all."""
        db = SessionLocal()
        try:
            if model_name:
                usage = db.query(ApiUsage).filter(ApiUsage.model_name == model_name).first()
                if usage:
                    usage.minute_requests = 0
                    usage.minute_tokens = 0
                    usage.day_requests = 0
                    # Don't reset last_request_at - it will be set on next request
                    db.commit()
                    logger.info(f"Manually reset quota for {model_name}")
                else:
                    logger.warning(f"No ApiUsage record found for {model_name}")
            else:
                # Reset all
                usages = db.query(ApiUsage).all()
                for usage in usages:
                    usage.minute_requests = 0
                    usage.minute_tokens = 0
                    usage.day_requests = 0
                db.commit()
                logger.info(f"Manually reset quota for all models ({len(usages)} records)")
        finally:
            db.close()

    def get_daily_usage(self, model_name: str) -> int:
        """Helper for dashboard health status."""
        db = SessionLocal()
        try:
            usage = self._get_usage(db, model_name)
            # Reset if needed before returning
            self._reset_if_needed(db, usage)
            db.commit()
            return usage.day_requests
        finally:
            db.close()

    def get_total_daily_usage(self) -> int:
        """Helper to sum usage across all models for the dashboard quota tracker."""
        db = SessionLocal()
        try:
            now = self._get_now()
            # Only sum requests from today
            usages = db.query(ApiUsage).all()
            total = 0
            for u in usages:
                if u.last_request_at is not None:
                    # Ensure naive comparison
                    last_req = u.last_request_at
                    if last_req.tzinfo is not None:
                        last_req = last_req.replace(tzinfo=None)
                    if now.date() == last_req.date():
                        total += u.day_requests
            return total
        finally:
            db.close()

    def check_yfinance_quota(self) -> bool:
        """Check if we can make a yfinance request (throttled)."""
        status = self.check_quota("yfinance")
        return status.get("can_call", False)

    def log_yfinance_usage(self):
        """Log a yfinance API call."""
        self.log_usage("yfinance", tokens=0)

# Module-level singleton
quota_svc = QuotaService()
