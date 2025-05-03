# hate-speech-detection-model


# Hate Speech Detection with DistilBERT

This project implements a hate speech detection model using DistilBERT to classify tweets as "hate," "offensive," or "neutral" based on the Hate Speech and Offensive Language Dataset. The model is trained in Google Colab with oversampling and class weights to handle dataset imbalance, achieving ~89.89% accuracy.

## Dataset

- **Source:** Hate Speech and Offensive Language Dataset.
- **Size:** 24,783 tweets (original), 46,833 after oversampling.

### Classes (original):
- **Hate (class 0):** ~5.76% (1,430 examples).
- **Offensive (class 1):** ~77.37% (19,190 examples).
- **Neutral (class 2):** ~16.78% (4,163 examples).

### Classes (oversampled):
- **Hate:** ~9.16% (4,290 examples).
- **Offensive:** ~81.95% (38,380 examples).
- **Neutral:** ~8.89% (4,163 examples).

- **File:** `labeled_data.csv` with columns `tweet` (text) and `class` (label).

## Requirements

- **Google Colab with GPU** (free tier).
- **Python libraries:**
  - `pandas`
  - `scikit-learn`
  - `transformers==4.51.3`
  - `torch`
  - `numpy`

## Setup

1. **Open Google Colab:**
   - Create a new notebook or use an existing one.
   - Enable GPU: `Runtime > Change runtime type > Hardware accelerator > GPU`.

2. **Install Dependencies:**
   ```python
   !pip install pandas scikit-learn transformers==4.51.3 torch numpy
   ```

3. **Upload Dataset:**

   * Download `labeled_data.csv` from the dataset repository.
   * Upload to Colab:

   ```python
   from google.colab import files
   uploaded = files.upload()  # Select labeled_data.csv
   ```

4. **Verify:**

   ```bash
   !ls /content/
   ```

5. **Clone or Upload Code:**

   * Copy `hate_speech_detection.py` and `test_hate_speech.py` into Colab cells or clone from this repository:

   ```bash
   !git clone <repository-url>
   ```

## Approach

The model addresses the dataset’s class imbalance using DistilBERT with oversampling and class weights.

### Preprocessing

* **Cleaning:** Remove URLs, mentions (`@user`), hashtags (`#tag`), and convert text to lowercase.

* **Oversampling:**

  * “Hate” examples repeated 3x (\~4,290 total).
  * “Offensive” examples repeated 1x (total 2x, \~38,380).
  * **Total dataset:** 46,833 examples.

### Model

* **Architecture:** DistilBERT (`distilbert-base-uncased`) with 3 output classes and 0.4 dropout.
* **Class Weights:** `[3.5, 2.5, 0.2]` to prioritize “hate” (3.5), boost “offensive” (2.5), and suppress “neutral” (0.2).

#### Training:

* **Epochs:** 3

* **Batch size:** 16

* **Learning rate:** 3e-5

* **Warmup steps:** 500

* **Weight decay:** 0.02

* **Optimizer:** AdamW (default in transformers)

* **Training steps:** \~7,026

* **Evaluation:** Weighted accuracy, F1, precision, recall per epoch.

## Implementation

### Main Script: `hate_speech_detection.py`

* Loads and preprocesses `labeled_data.csv`.
* Oversamples “hate” (3x) and “offensive” (2x).
* Splits data (80% train, 20% validation).
* Trains DistilBERT with a custom `WeightedTrainer` for class weights.
* Saves model to `/content/model_output`.
* Tests 3 sample tweets.

### Test Script: `test_hate_speech.py`

* Tests 13 additional tweets using the trained model.
* Outputs predictions and probabilities.

## Usage

### Run the Main Script:

1. Copy `hate_speech_detection.py` into a Colab cell.
2. Execute (\~15-20 min with GPU).

**Outputs:**

* Dataset size and class distribution.
* Training progress with metrics.
* Predictions for 3 test tweets.

**Example Output:**

```plaintext
Using transformers version: 4.51.3
Loading data...
Dataset size: 46833 examples
Class distribution:
label
1    0.819508  # offensive
0    0.091602  # hate
2    0.088890  # neutral
Name: proportion, dtype: float64
Training model...
[7026/7026 22:38, Epoch 3/3]
Epoch  Training Loss  Validation Loss  Accuracy  F1        Precision  Recall
1      0.268000       0.314208         0.880218  0.874572  0.881939   0.880218
2      0.226700       0.228042         0.899114  0.898462  0.906322   0.899114
3      0.215800       0.221393         0.898900  0.899785  0.910784   0.898900

Testing predictions:
Probabilities: hate=0.78, offensive=0.20, neutral=0.02
Text: I hate everyone in this group!
Prediction: hate

Probabilities: hate=0.32, offensive=0.64, neutral=0.04
Text: You're such a loser, go away.
Prediction: offensive

Probabilities: hate=0.01, offensive=0.36, neutral=0.62
Text: Have a nice day everyone!
Prediction: neutral
```

### Run the Test Script:

1. Copy `test_hate_speech.py` into a Colab cell.
2. Execute to test 13 tweets.

**Example Output:**

```plaintext
Probabilities: hate=0.78, offensive=0.20, neutral=0.02
Text: I hate everyone in this group!
Prediction: hate
[Additional predictions follow]
```

### View Screenshots:

Screenshots of outputs (e.g., metrics, predictions) are in the `/screenshots` folder in the repository: `screenshots/`.

## Notes

* **Performance:** Achieves \~89.89% accuracy and \~89.98% F1, with correct predictions for “hate” and “offensive” texts. The “neutral” class probability for positive tweets (e.g., 0.62 for “Have a nice day everyone!”) is lower than ideal, possibly due to dataset noise.

### Potential Improvements:

* Increase weights (e.g., `[4.0, 3.0, 0.1]`) to further suppress “neutral”.
* Clean dataset to reduce label noise in “offensive” and “neutral” examples.
* Use BERT or focal loss for better imbalance handling.

### Troubleshooting:

* **FileNotFoundError:** Verify `/content/labeled_data.csv` exists.
* **CUDA Out of Memory:** Reduce `per_device_train_batch_size` to 8.
* **Incorrect Predictions:** Check probabilities and adjust weights or oversampling.

## License

This project is licensed under the MIT License.

## Acknowledgments

* **Dataset:** (https://github.com/t-davidson/hate-speech-and-offensive-language)
* **Model:** Hugging Face Transformers.
