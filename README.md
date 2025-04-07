# 📧 Email Spam Classifier using Machine Learning

This project is a simple **Email Spam Classifier** built using Python and `scikit-learn`. It uses **TF-IDF vectorization** and **Logistic Regression** to classify emails as **spam** or **not spam** (ham).

---

## 🚀 Features

- Loads and processes email text data.
- Uses TF-IDF to convert email content into numerical features.
- Trains a logistic regression model to classify emails.
- Predicts and evaluates accuracy on both training and test data.
- Allows for testing custom email messages.

---

## 📁 Dataset

The dataset used should be a CSV file named `mail_data.csv` with at least the following columns:

- `Category` - email label: either `spam` or `ham`.
- `Message` - the text content of the email.

> Example CSV snippet:
> | Category | Message |
> |----------|---------|
> | ham | I'm going to be home soon. |
> | spam | Congratulations! You've won a prize! Click here... |

You can use public datasets from [Kaggle](https://www.kaggle.com) or the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.html).

---

## 🛠️ Installation & Setup

1. Clone the repo or download the project files.
2. Create a virtual environment (optional but recommended):

```bash
python -m venv .venv
```
