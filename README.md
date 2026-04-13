# HR Analytics Project (IBM Employee Attrition)

This project performs end-to-end HR analytics using the IBM HR Attrition dataset.

## What it does

- Loads data from `data.csv`
- Cleans data (missing values, duplicates, Attrition encoding)
- Performs EDA and prints statistics
- Creates required visualizations
- Prints key insights
- Trains a Logistic Regression model and prints accuracy
- Suggests optional improvements

## Files

- `data.csv` - input dataset
- `hr_analytics_project.py` - main project script
- `requirements.txt` - Python dependencies

## Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the project:

```bash
python hr_analytics_project.py
```

## Expected output

- Printed dataset information and EDA stats
- Visual plots:
  - Attrition countplot
  - Department vs attrition countplot
  - MonthlyIncome vs attrition boxplot
  - Age histogram
- Printed key HR insights
- Printed Logistic Regression accuracy and classification report
