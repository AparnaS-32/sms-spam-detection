# SMS Spam Detection using Machine Learning

## Project Overview
This project uses Machine Learning and Natural Language Processing (NLP) to classify SMS messages as **Spam** or **Ham (Normal)**.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Matplotlib
- TF-IDF Vectorization
- Multinomial Naive Bayes

## Dataset
- SMS Spam Collection Dataset
- Total messages: 5,572
- Ham messages: 4,825
- Spam messages: 747

## Project Workflow
1. Load and explore the SMS dataset.
2. Split the dataset into training and testing sets.
3. Convert text messages into numerical features using TF-IDF.
4. Train a Multinomial Naive Bayes model.
5. Evaluate model performance.
6. Test custom SMS messages.

## Model Performance
- Accuracy: **97.04%**

## Confusion Matrix
The project generates a confusion matrix visualization saved as `confusion_matrix.png`.

## How to Run
1. Install the required libraries:
   `py -m pip install pandas scikit-learn matplotlib`
2. Run the project:
   `py main.py`

## Author
Aparna S

## Note
This is an educational machine learning project. Model performance may vary with new datasets and real-world messages.
