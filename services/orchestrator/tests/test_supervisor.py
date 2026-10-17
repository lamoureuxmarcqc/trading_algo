"""Tests du superviseur."""
from app.supervisor import classify_task, route


def test_classify_finance():
    assert classify_task("Analyse mon portefeuille d'actions") == "finance"


def test_classify_health():
    assert classify_task("J'ai des symptomes medicaux") == "health"


def test_classify_research_default():
    assert classify_task("Trouve moi des articles") == "research"


def test_route_returns_agent():
    assert route("Investir dans AAPL") == "finance"
