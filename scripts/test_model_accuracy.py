import json
import pytest
import os

# ✅ Minimum required accuracy for registration
ACCURACY_THRESHOLD = 0.85

@pytest.mark.order(1)  # Optional: ensures this runs before registration tests if you want
def test_model_accuracy_threshold():
    """
    ✅ Test that ensures the model's test accuracy meets the required threshold
    before allowing registration.
    This reads from 'experiment_info.json' which is generated during evaluation.
    """
    assert os.path.exists("experiment_info.json"), (
        "❌ 'experiment_info.json' not found. "
        "Make sure the evaluation step ran before this test."
    )

    with open("experiment_info.json", "r") as f:
        model_info = json.load(f)

    accuracy = model_info.get("accuracy")
    assert accuracy is not None, (
        "❌ Accuracy not found in experiment_info.json. "
        "Ensure evaluation step saves it correctly."
    )

    assert accuracy >= ACCURACY_THRESHOLD, (
        f"❌ Model accuracy {accuracy:.2%} is below required threshold "
        f"({ACCURACY_THRESHOLD:.0%}). Registration should be skipped."
    )

    print(f"✅ Model passed accuracy test with {accuracy:.2%} accuracy")
