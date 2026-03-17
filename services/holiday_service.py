import logging
from datetime import datetime, date, timedelta
import pytz

logger = logging.getLogger(__name__)

class HolidayService:
    """
    Tracks NSE trading holidays for 2026 and provides alerts.
    """
    # NSE Holidays 2026 (Weekdays only)
    HOLIDAYS_2026 = {
        "2026-01-26": "Republic Day",
        "2026-03-03": "Holi",
        "2026-03-26": "Shri Ram Navami",
        "2026-03-31": "Shri Mahavir Jayanti",
        "2026-04-03": "Good Friday",
        "2026-04-14": "Dr. Baba Saheb Ambedkar Jayanti",
        "2026-05-01": "Maharashtra Day",
        "2026-05-28": "Bakri Id (Eid-ul-Adha)",
        "2026-06-26": "Muharram",
        "2026-08-28": "Id-e-Milad",
        "2026-10-02": "Mahatma Gandhi Jayanti",
        "2026-10-21": "Dussehra",
        "2026-11-24": "Guru Nanak Jayanti",
        "2026-12-25": "Christmas",
    }

    def __init__(self):
        self.ist = pytz.timezone('Asia/Kolkata')

    def is_holiday(self, check_date: date = None) -> tuple[bool, str]:
        """Check if a date is a trading holiday or weekend."""
        if check_date is None:
            check_date = datetime.now(self.ist).date()
        
        # Weekend Check
        if check_date.weekday() >= 5: # Saturday=5, Sunday=6
            return True, "Weekend (Market Closed)"
            
        # Holiday List Check
        date_str = check_date.strftime("%Y-%m-%d")
        if date_str in self.HOLIDAYS_2026:
            return True, self.HOLIDAYS_2026[date_str]
            
        return False, ""

    def get_upcoming_holiday(self) -> dict:
        """Find the next upcoming holiday within the next 7 days."""
        today = datetime.now(self.ist).date()
        for i in range(1, 8):
            future_date = today + timedelta(days=i)
            is_h, name = self.is_holiday(future_date)
            if is_h and future_date.weekday() < 5: # Only report weekday holidays as 'upcoming'
                return {
                    "date": future_date.strftime("%Y-%m-%d"),
                    "name": name,
                    "days_away": i
                }
        return None

holiday_svc = HolidayService()
