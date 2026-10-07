from click.testing import CliRunner

from cli import cli


def test_cli_lists_download_corpus_command():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "download-corpus" in result.output


def test_download_corpus_help_mentions_freeze():
    result = CliRunner().invoke(cli, ["download-corpus", "--help"])
    assert result.exit_code == 0
    assert "--freeze" in result.output
