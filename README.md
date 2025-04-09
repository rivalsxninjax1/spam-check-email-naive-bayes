# SMS Spam Classifier Using Multinomial Naive Bayes

This project demonstrates a real-life application of the Naive Bayes algorithm by developing an SMS spam classifier. The classifier distinguishes between spam and ham messages using a bag-of-words model and a Multinomial Naive Bayes classifier.

## Dataset

The dataset (`sms_spam_collection.csv`) contains SMS messages along with their labels (`spam` or `ham`). The dataset is a subset inspired by the SMS Spam Collection dataset.

### Example Data

| Label | Message |
|-------|---------|
| ham   | I'm gonna be home soon and i don't want to talk... |
| spam  | WINNER!! As a valued network customer you have been selected... |
| ...   | ... |

## Features

- **Text Preprocessing:** Using `CountVectorizer` to convert messages into a bag-of-words model.
- **Classification:** Implemented using the Multinomial Naive Bayes classifier from scikit-learn.
- **Evaluation:** The model is evaluated using accuracy scores and a detailed classification report.

## How to Run

1. **Clone the repository or download the files:**

    ```bash
    git clone https://github.com/your-username/sms-spam-classifier.git
    cd sms-spam-classifier
    ```

2. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Notebook/Script:**

    - For Jupyter Notebook:
    
      ```bash
      jupyter lab sms_spam_classifier.ipynb
      ```

    - For Python script:
    
      ```bash
      python sms_spam_classifier.py
      ```

## Requirements

See `requirements.txt` for the dependencies.

## License

This project is open source and available under the [MIT License](LICENSE).

# spam-check-email-naive-bayes
