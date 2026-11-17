import pytest

from zeroctx.router import cosine, route


def test_cosine_basic():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_route_steers_above_threshold():
    topics = {"a": [1.0, 0.0], "b": [0.0, 1.0]}
    decision = route([0.9, 0.1], topics, threshold=0.6)
    assert decision.topic_id == "a"
    assert decision.score is not None and decision.score > 0.6


def test_route_new_topic_below_threshold():
    topics = {"a": [1.0, 0.0]}
    decision = route([0.0, 1.0], topics, threshold=0.6)
    assert decision.topic_id is None
    assert decision.score == pytest.approx(0.0)


def test_route_threshold_is_inclusive():
    decision = route([1.0, 0.0], {"a": [1.0, 0.0]}, threshold=1.0)
    assert decision.topic_id == "a"


def test_route_without_active_topics():
    decision = route([1.0, 0.0], {}, threshold=0.6)
    assert decision.topic_id is None
    assert decision.score is None
