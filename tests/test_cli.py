import pytest

from film.cli import main


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "film 0.1.0" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out


def test_main_sets_ssl_cert_file(monkeypatch):
    import os

    monkeypatch.delenv("SSL_CERT_FILE", raising=False)
    main(["doctor"])
    assert os.path.isfile(os.environ["SSL_CERT_FILE"])


@pytest.mark.parametrize("cmd", ["ingest", "run"])
def test_profile_flag_rejects_unknown_names(cmd, capsys):
    from film.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args([cmd, "x.mp4", "--profile", "hudl"])
    assert "invalid choice" in capsys.readouterr().err


@pytest.mark.parametrize("cmd", ["ingest", "run"])
def test_profile_flag_accepts_known_names(cmd):
    from film.cli import build_parser

    assert build_parser().parse_args([cmd, "x.mp4", "--profile", "hudl-tactical"]).profile == "hudl-tactical"
