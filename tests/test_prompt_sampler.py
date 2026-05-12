from reps.config import PromptConfig
from reps.prompt_sampler import PromptSampler


def test_inspiration_metric_features_format_fragment_placeholders():
    sampler = PromptSampler(PromptConfig())

    features = sampler._extract_unique_features(
        {
            "id": "candidate",
            "code": "",
            "metrics": {
                "combined_score": 0.95,
                "validity": 0.2,
            },
        }
    )

    assert "[Fragment formatting error:" not in features
    assert "Excellent combined_score (0.950)" in features
    assert "Alternative validity approach" in features
