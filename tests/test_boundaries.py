import pytest
from pydantic import ValidationError

from src.app.dependencies import AppSettings
from src.evaluation.eval_ragas import evaluate
from src.rag.ingestion import load_and_split


def test_paid_evaluation_requires_opt_in(tmp_path):
    with pytest.raises(ValueError, match="opt-in"):
        evaluate(tmp_path / "unused.json")


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/v1/chat/completions",
        "http://localhost:8081/v1/chat/completions",
        "http://127.0.0.1:8081/other",
    ],
)
def test_language_model_url_must_be_fixed_loopback(url):
    with pytest.raises(ValidationError):
        AppSettings(local_llm_url=url)


def test_oversize_file_rejected(tmp_path):
    (tmp_path / "large.txt").write_text("x" * 1_000_001)
    with pytest.raises(ValueError, match="size budget"):
        load_and_split(tmp_path, allowed_root=tmp_path)
