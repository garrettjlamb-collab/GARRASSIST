from film.interpret.fuse import fuse


def S(made, missed, none):
    return {"made": made, "missed": missed, "none": none}


def test_two_classifiers_that_agree_and_are_sure_give_a_strong_call():
    out = fuse(S(0.8, 0.15, 0.05), S(0.9, 0.05, 0.05))
    assert out["label"] == "made" and out["agreement"] == "strong" and abs(out["prob"] - 0.85) < 1e-6
    assert out["path"][0] == "made" and out["pictures"][0] == "made"


def test_agreeing_without_much_conviction_is_only_fair_or_weak():
    assert fuse(S(0.60, 0.30, 0.10), S(0.62, 0.28, 0.10))["agreement"] == "fair"      # 0.68 made among made and missed
    assert fuse(S(0.45, 0.40, 0.15), S(0.47, 0.43, 0.10))["agreement"] == "weak"


def test_disagreeing_on_made_or_missed_is_a_split_whatever_the_average_says():
    out = fuse(S(0.9, 0.05, 0.05), S(0.2, 0.7, 0.1))
    assert out["agreement"] == "split" and out["label"] == "made"


def test_a_likely_non_shot_is_never_a_strong_call():
    out = fuse(S(0.6, 0.05, 0.35), S(0.6, 0.05, 0.35))
    assert out["label"] == "made" and out["agreement"] == "fair"                        # sure of made over missed, unsure it is a shot
    assert fuse(S(0.1, 0.1, 0.8), S(0.2, 0.1, 0.7))["label"] == "none"


def test_one_classifier_alone_has_no_agreement_to_report():
    out = fuse(S(0.8, 0.15, 0.05), None)
    assert out["label"] == "made" and out["prob"] == 0.8 and out["agreement"] is None and out["pictures"] is None
