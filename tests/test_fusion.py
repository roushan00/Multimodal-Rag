from app.retrieve.fusion import dedupe_keep_order, rrf


def test_rrf_rewards_agreement():
    text = ["a", "b", "c"]
    visual = ["c", "a", "d"]
    ranked = [x for x, _ in rrf([text, visual])]
    assert ranked[0] == "a"            # rank 1 + rank 2 beats everything
    assert set(ranked) == {"a", "b", "c", "d"}


def test_rrf_weights():
    ranked = [x for x, _ in rrf([["a"], ["b"]], weights=[1.0, 2.0])]
    assert ranked == ["b", "a"]


def test_dedupe():
    assert dedupe_keep_order([1, 2, 1, 3, 2]) == [1, 2, 3]
