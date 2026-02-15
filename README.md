# Chess-Games-Outcome-Predictor

## Project Overview

This project is a Chess Game Result Prediction system that uses supervised Machine Learning algorithms to predict the outcome of a chess game (Win / Loss / Draw) based on game-related features such as rating differences, number of moves, opening type, and other statistical parameters.

The system analyzes historical chess data and applies classification algorithms to determine the most likely result of a match.

Key Features:
- **Multi-Class Classification**: Predicts game outcome (White Win, Black Win, or Draw).

- **Multiple ML Models**: Implements Logistic Regression, Decision Tree, Random Forest (or the models you used).

- **Performance Evaluation**: Includes accuracy score, classification report, and confusion matrix.

- **Data Visualization**: Displays important graphs such as feature importance, class distribution, and confusion matrix.

- **Modular Notebook Structure**: Clean separation of preprocessing, modeling, and evaluation logic.

---

## Tech Stack & Algorithms

- **Language**: Python - Core programming language.
- **ML Libraries**: Scikit-learn - Implementing classification algorithms.
- **Data Handling**: Pandas, NumPy - Data loading, preprocessing, and feature engineering.
- **Visualization**: Matplotlib, Seaborn - Generating performance and distribution graphs.
- **Notebook Environment**: Jupyter Notebook - Development and experimentation.
- **Models Used**: 1. Logistic Regression, 2. Decision Tree Classifier, 3. Random Forest Classifier

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/GaganMS004/Chess-Games-Result-Outcome-Predictor.git
   ```
2. Install Python libraries:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
3. Data Setup: Ensure your dataset file (for example: games.csv) is placed in the project’s root directory.

---

## How to Run the Application

This project is structured within a single integrated notebook/script.

1. Open the notebook or Python file.

2. Run all cells sequentially (or execute the script).

3. The program will:

- Load and preprocess the dataset.

- Perform feature engineering (e.g., rating difference).

- Split data into training, validation, and test sets.

- Train Logistic Regression, Decision Tree, and Random Forest models.

- Evaluate performance using accuracy score and classification report.

- Display performance comparison graphs.

---

## Using the Project

1. Run the notebook.

2. Train the models using the training dataset.

3. Evaluate model performance using validation and test datasets.

4. Observe visualizations to compare model accuracy and interpret feature importance.

5. Use the trained model to predict the result of new chess game data.

---

## Future Enhancements
- Implement Cross-Validation for better generalization.

- Apply Hyperparameter Tuning using GridSearchCV.

- Deploy the model using Flask or Streamlit.

- Integrate live chess data from platforms like Lichess or Chess.com.

- Add advanced analytics such as opening performance trends.
