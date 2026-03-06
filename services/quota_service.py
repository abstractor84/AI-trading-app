import logging
from datetime import datetime, timedelta, timezone, UTC
from sqlalchemy.orm import Session
from database import SessionLocal
from models import ApiUsage
import os

logger = logging.getLogger(__name__)

class QuotaService:
    def __init__(self):
        # Strict user limits: max 30/day per provider
        self.defaults = {
            "google": {"rpm": 15, "tpm": 1000000, "rpd": 30},
            "groq": {"rpm": 30, "tpm": 144000, "rpd": 30},
            "sambanova": {"rpm": 15, "tpm": 50000, "rpd": 30},
        }


    def _get_usage(self, db: Session, model_name: str) -> ApiUsage:
        usage = db.query(ApiUsage).filter(ApiUsage.model_name == model_name).first()
        if not usage:
            limits = self.defaults.get(model_name, {"rpm": 5, "tpm": 250000, "rpd": 20})
            usage = ApiUsage(
                model_name=model_name,
                limit_rpm=limits["rpm"],
                limit_tpm=limits["tpm"],
                limit_rpd=limits["rpd"]
            )
            db.add(usage)
            db.commit()
            db.refresh(usage)
        return usage

    def check_quota(self, model_name: str) -> dict:
        """Check if we have enough quota for 1 request."""
        db = SessionLocal()
        try:
            usage = self._get_usage(db, model_name)
            now = datetime.now(UTC).replace(tzinfo=None)


            # Reset minute counters if a minute has passed
            if (now - usage.last_request_at) > timedelta(minutes=1):
                usage.minute_requests = 0
                usage.minute_tokens = 0
            
            # Reset day counters if it's a new calendar day
            if now.date() > usage.last_request_at.date():
                usage.day_requests = 0

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
            db.commit()
            return status
        finally:
            db.close()

    def log_usage(self, model_name: str, tokens: int = 0):
        """Update usage after a successful call."""
        db = SessionLocal()
        try:
            usage = self._get_usage(db, model_name)
            now = datetime.now(UTC).replace(tzinfo=None)


            # Reset minute counters if a minute has passed
            if (now - usage.last_request_at) > timedelta(minutes=1):
                usage.minute_requests = 0
                usage.minute_tokens = 0
            
            # Reset day counters if it's a new calendar day
            if now.date() > usage.last_request_at.date():
                usage.day_requests = 0

            usage.minute_requests += 1
            usage.day_requests += 1
            usage.minute_tokens += tokens
            usage.last_request_at = now
            
            db.commit()
            logger.info(f"Logged quota for {model_name}: +1 req, +{tokens} tokens. Day total: {usage.day_requests}")
        finally:
            db.close()

    def get_daily_usage(self, model_name: str) -> int:
        """Helper for dashboard health status."""
        db = SessionLocal()
        try:
            usage = self._get_usage(db, model_name)
            return usage.day_requests
        finally:
            db.close()

    def get_total_daily_usage(self) -> int:
        """Helper to sum usage across all models for the dashboard quota tracker."""
        db = SessionLocal()
        try:
            now = datetime.now(UTC).replace(tzinfo=None)
            # Only sum requests from today
            usages = db.query(ApiUsage).all()
            total = 0
            for u in usages:
                if now.date() == u.last_request_at.date():
                    total += u.day_requests
            return total
        finally:
            db.close()

# Module-level singleton
quota_svc = QuotaService()
