from pathlib import Path

from film.cli import build_parser


def test_review_command_registered_and_page_has_the_tabs():
    args = build_parser().parse_args(["review", "--port", "9999", "--no-open"])
    assert args.port == 9999 and args.no_open is True and args.func.__module__ == "film.commands.review"
    html = (Path("film") / "review" / "static" / "index.html").read_text()
    for needle in ('data-tab="shots"', 'data-tab="rim"', 'data-tab="ball"', 'data-tab="rimclf"', "/api/games", "Confirm all above 0.9"):
        assert needle in html
