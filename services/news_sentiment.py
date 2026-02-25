import requests
from bs4 import BeautifulSoup
import logging
from google import genai
import os
import json

logger = logging.getLogger(__name__)

class NewsSentimentService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None
            logger.warning("GEMINI_API_KEY not found. Sentiment will be NEUTRAL.")

    def fetch_news(self, ticker: str) -> list[str]:
        """Fetch latest news headlines for the ticker using Google News RSS."""
        clean_ticker = ticker.replace(".NS", "")
        url = f"https://news.google.com/rss/search?q={clean_ticker}+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en"
        try:
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.content, features="xml")
            items = soup.find_all("item")
            headlines = [item.title.text for item in items[:5]] # top 5 recent headlines
            return headlines
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return []

    def score_sentiment(self, headlines: list[str]) -> dict:
        """Use Gemini to score sentiment of headlines as POSITIVE, NEGATIVE, or NEUTRAL."""
        if not headlines:
            return {"sentiment": "NEUTRAL", "reason": "No news found"}
        
        if not self.client:
            return {"sentiment": "NEUTRAL", "reason": "No API key config"}

        prompt = f"""
        Analyze the following recent news headlines for an Indian stock and determine the overall sentiment.
        Headlines: {headlines}
        
        Respond ONLY with a JSON object in this exact format:
        {{"sentiment": "POSITIVE|NEGATIVE|NEUTRAL", "reason": "Short 1-sentence explanation"}}
        """
        
        try:
            # Try Gemini 3.0 Flash first
            model_name = 'gemini-3.0-flash'
            try:
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )
            except Exception as e:
                logger.warning(f"Gemini 3.0 Flash failed, falling back to 2.5: {e}")
                model_name = 'gemini-2.5-flash'
                response = self.client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1
                    )
                )
            data = json.loads(response.text)
            return data
        except Exception as e:
            logger.error(f"Error calling Gemini for sentiment: {e}")
            return {"sentiment": "NEUTRAL", "reason": "Error parsing sentiment"}
