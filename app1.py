from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import pandas as pd

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # For session handling

# Load your trained model
model = joblib.load('best_model_Random Forest.joblib')

# Categories for encoding (must match your training)
employment_type_categories = ['Salaried', 'Student', 'Self-employed', 'Retired']
financial_goal_categories = ['Wealth Creation', 'Tax Saving', 'Child Education', 'Emergency', 'Retirement']
tax_bracket_categories = ['>15L', '<5L', '5–10L', '10–15L']
investment_knowledge_categories = ['Intermediate', 'Beginner', 'Expert']

# Investment product descriptions with emojis
product_descriptions = {
    "PPF": "🟢 PPF (Public Provident Fund): A safe, long-term government savings scheme with tax benefits.",
    "ELSS": "🟠 ELSS (Equity Linked Saving Scheme): A tax-saving mutual fund with market exposure and a 3-year lock-in period.",
    "FD": "🔵 Fixed Deposit: A low-risk, fixed-return investment option with flexible tenure.",
    "ULIP": "🟣 ULIP (Unit Linked Insurance Plan): Combines insurance and investment, eligible for tax deductions.",
    "NPS": "🟤 NPS (National Pension Scheme): Retirement-focused investment with tax-saving advantages.",
    "SSY": "🧡 Sukanya Samriddhi Yojana: Designed for girl child education/savings, offering attractive interest rates."
}

# If your model returns numbers, use this decoder:
label_decoder = {
    0: "ELSS",
    1: "PPF",
    2: "FD",
    3: "ULIP",
    4: "NPS",
    5: "SSY"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form inputs
            age = int(request.form['age'])
            income = float(request.form['annual_income'])
            employment = request.form['employment_type']
            horizon = float(request.form['investment_horizon'])
            risk_score = float(request.form['risk_tolerance_score'])
            goal = request.form['financial_goal']
            tax = request.form['tax_bracket']
            knowledge = request.form['investment_knowledge']

            # Encode categorical inputs
            encoded_data = pd.DataFrame([[
                age,
                income,
                horizon,
                risk_score,
                employment_type_categories.index(employment),
                financial_goal_categories.index(goal),
                tax_bracket_categories.index(tax),
                investment_knowledge_categories.index(knowledge)
            ]], columns=[
                'num_pipeline__age',
                'num_pipeline__annual_income',
                'num_pipeline__investment_horizon',
                'num_pipeline__risk_tolerance_score',
                'cat_pipeline__employment_type',
                'cat_pipeline__financial_goal',
                'cat_pipeline__tax_bracket',
                'cat_pipeline__investment_knowledge'
            ])

            # Predict
            prediction = model.predict(encoded_data)[0]

            # Convert index to label (if numeric) then to description
            label = label_decoder.get(prediction, prediction)  # If already string, skip decoding
            result = product_descriptions.get(label, f"Recommended: {label}")

            # Pass to result page
            session['result'] = result
            return redirect(url_for('result'))

        except Exception as e:
            session['result'] = f"❌ Error: {str(e)}"
            return redirect(url_for('result'))

    return render_template('predict.html')

@app.route('/result')
def result():
    return render_template('result.html', result=session.get('result'))

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
