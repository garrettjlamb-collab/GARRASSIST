from __future__ import annotations


def register(sub) -> None:
    parser = sub.add_parser("review", help="serve the review and labeling page in your browser")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="do not open the browser automatically")
    parser.set_defaults(func=run)


def run(args) -> int:
    import threading
    import webbrowser

    import uvicorn

    from film.review.app import create_app

    url = f"http://127.0.0.1:{args.port}/"
    if not args.no_open:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"review page at {url}  (Ctrl+C to stop)")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, log_level="warning")
    return 0
