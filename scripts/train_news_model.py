import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from user_data.data.ml_signal import DEFAULT_TRAINING_FILE, train_news_classifier


def main() -> None:
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRAINING_FILE
    metrics = train_news_classifier(csv_path)
    print("Trained news classifier")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
