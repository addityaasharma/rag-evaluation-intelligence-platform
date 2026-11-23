import pytest
from conftest import FakeLLM

from ragkit.models import Answer
from ragkit.verification import HallucinationDetector, VerificationError

CLAIMS = (
    '{"claims": [{"text": "Backups last 35 days", "supported": true},'
    ' {"text": "Backups are encrypted with RSA", "supported": false}]}'
)


def test_detector_flags_unsupported_claims(make_chunk, ranked):
    detector = HallucinationDetector(FakeLLM(CLAIMS))
    result = detector.check("answer", ranked("a"))
    assert result.has_hallucination
    assert [c.text for c in result.unsupported] == ["Backups are encrypted with RSA"]


def test_detector_with_all_claims_supported(ranked):
    detector = HallucinationDetector(FakeLLM('{"claims": [{"text": "x", "supported": true}]}'))
    assert not detector.check("answer", ranked("a")).has_hallucination


@pytest.mark.parametrize("raw", ["nope", '{"claims": "x"}', '{"claims": [{"text": "x"}]}'])
def test_detector_rejects_malformed_output(raw, ranked):
    with pytest.raises(VerificationError):
        HallucinationDetector(FakeLLM(raw)).check("answer", ranked("a"))


def test_cited_indices_ignore_out_of_range(ranked):
    answer = Answer(
        question="q", text="See [2] and [1], also [9] and [2].", contexts=ranked("a", "b")
    )
    assert answer.cited_indices == [1, 2]
