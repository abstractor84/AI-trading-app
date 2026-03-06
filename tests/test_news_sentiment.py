import pytest
import responses
from unittest.mock import patch, MagicMock
from services.news_sentiment import NewsSentimentService, _keyword_sentiment, _rss_fetch, _tavily_fetch, _ddgs_fetch

def test_keyword_sentiment_bullish():
    headlines = ["Stock surge on high profit", "Rally continue for bullish trend"]
    res = _keyword_sentiment(headlines)
    assert res["sentiment"] == "POSITIVE"
    assert res["score"] > 60

def test_keyword_sentiment_bearish():
    headlines = ["Stock crash on loss", "Drop in profit and downgrade"]
    res = _keyword_sentiment(headlines)
    assert res["sentiment"] == "NEGATIVE"
    assert res["score"] < 40

def test_keyword_sentiment_neutral():
    headlines = ["Stock jump drop", "Rise fall"] # 1 pos, 1 neg -> 1/2 = 50% -> Neutral
    res = _keyword_sentiment(headlines)
    assert res["sentiment"] == "NEUTRAL"
    assert res["score"] == 50


@responses.activate
def test_rss_fetch_success():
    ticker = "RELIANCE"
    url = f"https://news.google.com/rss/search?q={ticker}+NSE+stock&hl=en-IN&gl=IN&ceid=IN:en"
    xml_content = """<rss><channel><item><title>Reliance Profit Jump</title><link>http://link</link></item></channel></rss>"""
    responses.add(responses.GET, url, body=xml_content, status=200)
    
    res = _rss_fetch(ticker)
    assert len(res) == 1
    assert res[0]["title"] == "Reliance Profit Jump"

@responses.activate
def test_rss_fetch_failure():
    responses.add(responses.GET, "https://news.google.com/rss/search", status=500)
    res = _rss_fetch("FAIL")
    assert res == []

def test_tavily_fetch_no_key():
    with patch("os.getenv", return_value=None):
        assert _tavily_fetch("query") == []

def test_ddgs_fetch_success():
    with patch("duckduckgo_search.DDGS") as mock_ddgs:
        mock_instance = mock_ddgs.return_value.__enter__.return_value
        mock_instance.news.return_value = [{"title": "DDGS News", "link": "http://ddgs"}]
        res = _ddgs_fetch("query")
        assert res[0]["title"] == "DDGS News"

def test_fetch_news_fallback_rss_to_ddgs():
    svc = NewsSentimentService()
    with patch("services.news_sentiment._rss_fetch", return_value=[]), \
         patch("services.news_sentiment._ddgs_fetch", return_value=[{"title": "DDGS Result"}]):
        # Default engine is gemini/RSS
        res = svc.fetch_news("TICKER", search_engine="gemini", fallback=True)
        assert res[0]["title"] == "DDGS Result"

def test_score_sentiment_no_news():
    svc = NewsSentimentService()
    res = svc.score_sentiment([])
    assert res["sentiment"] == "NEUTRAL"

def test_score_sentiment_quota_exceeded():
    svc = NewsSentimentService()
    with patch("services.news_sentiment.quota_svc.check_quota", return_value={"can_call": False}):
        # Should fallback to keyword analysis
        res = svc.score_sentiment([{"title": "Market crash"}])
        assert res["sentiment"] == "NEGATIVE"

@patch("services.news_sentiment.genai.Client")
def test_call_google_success(mock_genai):
    svc = NewsSentimentService()
    svc.google_client = mock_genai.return_value
    mock_response = MagicMock()
    mock_response.text = '{"sentiment": "NEUTRAL", "reason": "Google ok"}'
    mock_response.usage_metadata.total_token_count = 100
    mock_genai.return_value.models.generate_content.return_value = mock_response

    
    res = svc._call_google("model", "prompt")
    assert res["sentiment"] == "NEUTRAL"

def test_tavily_fetch_success():
    with patch("services.news_sentiment.os.getenv", return_value="fake_key"), \
         patch("tavily.TavilyClient") as mock_tavily:
        mock_instance = mock_tavily.return_value
        mock_instance.search.return_value = {"results": [{"title": "Tavily News", "url": "http://tavily"}]}
        res = _tavily_fetch("query")
        assert res[0]["title"] == "Tavily News"

def test_fetch_news_all_engines():
    svc = NewsSentimentService()
    with patch("services.news_sentiment._tavily_fetch", return_value=[{"title": "T1"}]), \
         patch("services.news_sentiment._ddgs_fetch", return_value=[{"title": "D1"}]), \
         patch("services.news_sentiment._rss_fetch", return_value=[{"title": "R1"}]):
        
        assert svc.fetch_news("K1", search_engine="tavily")[0]["title"] == "T1"
        assert svc.fetch_news("K1", search_engine="ddgs")[0]["title"] == "D1"
        assert svc.fetch_news("K1", search_engine="gemini")[0]["title"] == "R1"

def test_fetch_news_fallback_cascade():
    svc = NewsSentimentService()
    # Test Tavily -> DDGS -> RSS cascade
    with patch("services.news_sentiment._tavily_fetch", return_value=[]), \
         patch("services.news_sentiment._ddgs_fetch", return_value=[]), \
         patch("services.news_sentiment._rss_fetch", return_value=[{"title": "RSS Last Resort"}]):
        res = svc.fetch_news("K1", search_engine="tavily", fallback=True)
        assert res[0]["title"] == "RSS Last Resort"

def test_score_sentiment_provider_fallback():
    svc = NewsSentimentService()
    # Unknown provider should fallback to keywords
    with patch("services.news_sentiment._keyword_sentiment") as mock_kw:
        svc.score_sentiment([{"title": "H1"}], provider="unknown")
        assert mock_kw.called

@patch("requests.post")
def test_score_sentiment_all_providers(mock_post):
    svc = NewsSentimentService()
    svc.groq_key = "k"
    svc.samba_key = "k"
    svc.google_key = "k"
    
    mock_res = MagicMock()
    mock_res.status_code = 200
    mock_res.json.return_value = {"choices": [{"message": {"content": '{"sentiment": "POSITIVE"}'}}], "usage": {}}
    mock_post.return_value = mock_res
    
    # Test Groq dispatch
    res = svc.score_sentiment([{"title": "H1"}], provider="groq")
    assert res["sentiment"] == "POSITIVE"
    
    # Test Samba dispatch
    res = svc.score_sentiment([{"title": "H1"}], provider="sambanova")
    assert res["sentiment"] == "POSITIVE"

def test_score_sentiment_exception_fallback():
    svc = NewsSentimentService()
    with patch.object(svc, "_call_google", side_effect=Exception("API Down")):
        res = svc.score_sentiment([{"title": "H1"}], provider="google")
        assert "reason" in res # Should fallback to keyword

def test_ddgs_fetch_fallback_and_error():
    # Test fallback to text() if news() empty
    with patch("duckduckgo_search.DDGS") as mock_ddgs:
        mock_instance = mock_ddgs.return_value.__enter__.return_value
        mock_instance.news.return_value = []
        mock_instance.text.return_value = [{"title": "Text Result", "href": "http://ddgs"}]
        assert _ddgs_fetch("q")[0]["title"] == "Text Result"

        
        # Test exception
        mock_instance.text.side_effect = Exception("DDGS Fail")
        assert _ddgs_fetch("q") == []

def test_google_client_none_error():
    svc = NewsSentimentService()
    svc.google_client = None
    with pytest.raises(ValueError, match="GEMINI_API_KEY missing"):
        svc._call_google("m", "p")

def test_samba_key_missing_error():
    svc = NewsSentimentService()
    svc.samba_key = None
    with pytest.raises(ValueError, match="SAMBA_API_KEY missing"):
        svc._call_sambanova("m", "p")

@responses.activate
def test_rss_exception_handling():
    with patch("services.news_sentiment.requests.get", side_effect=Exception("Timeout")):
        assert _rss_fetch("T") == []

def test_tavily_exception_handling():
    with patch("services.news_sentiment.os.getenv", return_value="key"), \
         patch("tavily.TavilyClient", side_effect=Exception("Tavily Error")):
        assert _tavily_fetch("q") == []

def test_fetch_news_empty_fallbacks():
    svc = NewsSentimentService()
    # Test DDGS empty -> RSS fallback
    with patch("services.news_sentiment._ddgs_fetch", return_value=[]), \
         patch("services.news_sentiment._rss_fetch", return_value=["RSS Fallback"]):
        assert svc.fetch_news("K", search_engine="ddgs", fallback=True) == ["RSS Fallback"]
    
    # Test RSS empty -> DDGS fallback
    with patch("services.news_sentiment._rss_fetch", return_value=[]), \
         patch("services.news_sentiment._ddgs_fetch", return_value=["DDGS Fallback"]):
        assert svc.fetch_news("K", search_engine="gemini", fallback=True) == ["DDGS Fallback"]

@patch("requests.post")
def test_call_sambanova_json_clean(mock_post):
    svc = NewsSentimentService()
    svc.samba_key = "k"
    res_mock = MagicMock()
    res_mock.status_code = 200
    res_mock.json.return_value = {"choices": [{"message": {"content": '```json\n{"sentiment": "NEUTRAL"}\n```'}}], "usage": {}}
    mock_post.return_value = res_mock
    res = svc._call_sambanova("m", "p")
    assert res["sentiment"] == "NEUTRAL"



