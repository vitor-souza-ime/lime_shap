
````markdown
# Comparative Analysis of LIME and SHAP for Explaining Sentiment Classification Models on Text Data

This project presents a comparative study between two widely used model-agnostic explainability techniques: **LIME (Local Interpretable Model-agnostic Explanations)** and **SHAP (SHapley Additive exPlanations)**. The evaluation is conducted using a sentiment classification model applied to simulated tweets labeled as positive or negative.

## 📄 Main Script

- `main01.py`: Complete Python script that performs data simulation, model training with XGBoost, local explanation using LIME and SHAP, fidelity and execution time measurement, and graphical visualization of results.

## 🧪 Features

- Simulates 100 tweets with binary sentiment labels (positive and negative).
- Trains a classifier using the XGBoost algorithm.
- Applies LIME and SHAP to explain model predictions.
- Calculates and compares two key metrics:
  - **Execution Time (seconds)**: Time required to generate a single explanation.
  - **Local Fidelity (MSE)**: Mean squared error between the model’s predicted probability and the explanation output.
- Displays comparative bar charts for time and fidelity per instance and on average.

## 🛠️ Technologies Used

- Python 3
- [scikit-learn](https://scikit-learn.org/)
- [XGBoost](https://xgboost.readthedocs.io/)
- [LIME](https://github.com/marcotcr/lime)
- [SHAP](https://github.com/shap/shap)
- matplotlib
- numpy

## 📦 Installation

It is recommended to use a virtual environment. Install dependencies with:

```bash
pip install numpy matplotlib scikit-learn xgboost shap lime
````

## ▶️ How to Run

Run the script using Python:

```bash
python main01.py
```

The program will output model accuracy, execution times, local fidelity scores, and display graphical comparisons between LIME and SHAP.

## 📊 Expected Results

* **LIME** typically takes longer to compute but often shows higher local fidelity.
* **SHAP** is significantly faster, especially when using models like XGBoost, with slightly lower fidelity in some cases.
* The charts generated at the end summarize the performance of both methods.

## 💡 About Model Explainability

Model explainability refers to the ability to interpret and understand the decisions made by a machine learning model. LIME and SHAP are local explanation methods that identify how each input feature contributed to a specific prediction, improving transparency and trust in AI systems.

## 📚 License

This project is open for educational and academic use.

