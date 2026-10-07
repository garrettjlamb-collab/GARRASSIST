"""Two opinions on a shot, combined: the ball's path (film.train.rimclf) and the pictures (film.train.picclf).

They err on different shots. Measured on six reviewed games, each held out in turn: averaged they are right
on 0.83 of made-versus-missed calls (0.74 and 0.81 alone); where they agree with conviction and the card
looks like a shot, 90% of cards are a real shot with the right call, covering about half of all real shots;
where they disagree it is a coin toss."""
from __future__ import annotations

CLASSES = ("made", "missed", "none")
STRONG_CONVICTION = 0.75        # share of made among made+missed (or of missed), averaged over the two
FAIR_CONVICTION = 0.65
STRONG_NONE_MAX = 0.25          # above this averaged chance of "nothing happened", a call is not strong


def _best(scores: dict) -> list:
    label = max(CLASSES, key=lambda c: scores[c])
    return [label, round(float(scores[label]), 4)]


def fuse(path_scores: dict, picture_scores: dict | None) -> dict:
    """The combined call. `agreement` is "strong", "fair", "weak" or "split"; None with only one opinion."""
    if picture_scores is None:
        label, prob = _best(path_scores)
        return {"label": label, "prob": prob, "agreement": None, "path": [label, prob], "pictures": None}
    avg = {c: (path_scores[c] + picture_scores[c]) / 2 for c in CLASSES}
    label, prob = _best(avg)
    same_side = (path_scores["made"] > path_scores["missed"]) == (picture_scores["made"] > picture_scores["missed"])
    shot_mass = avg["made"] + avg["missed"]
    conviction = max(avg["made"], avg["missed"]) / shot_mass if shot_mass > 0 else 0.0
    if not same_side:
        agreement = "split"
    elif conviction >= STRONG_CONVICTION and avg["none"] <= STRONG_NONE_MAX and label != "none":
        agreement = "strong"
    elif conviction >= FAIR_CONVICTION:
        agreement = "fair"
    else:
        agreement = "weak"
    return {"label": label, "prob": prob, "agreement": agreement, "path": _best(path_scores), "pictures": _best(picture_scores)}
