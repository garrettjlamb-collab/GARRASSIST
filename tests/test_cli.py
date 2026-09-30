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
