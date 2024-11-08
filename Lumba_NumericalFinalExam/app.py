# Lumba, Pierre Ellezar A.
# CS-302
# Python File
# Computations of User Inputs will be done in this File
import base64
from io import BytesIO
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__, template_folder='template')

@app.route('/')
def home_page():
    return render_template('home-page.html')

@app.route('/members')
def members_page():
    return render_template('members-page.html')

@app.route('/import_csv', methods=['POST'])
def import_csv():
    csv_file_path = request.form['import_csv']
    data = pd.read_csv(csv_file_path)
    return render_template('home-page.html', data=data.to_html())

@app.route('/analyze_data', methods=['POST'])
def analyze_data():
    dependent_var = request.form['input-dependent']
    independent_vars = request.form['input-independent'].split(',')
    data = pd.read_csv(request.files['csv_file'])

    x = data[independent_vars].apply(pd.to_numeric, errors='coerce')
    y = data[dependent_var].apply(pd.to_numeric, errors='coerce')

    # x = data[independent_vars]
    # y = data[dependent_var]
    # Linear Regression Model in Python
    model = LinearRegression()
    model.fit(x, y)

    predictions = model.predict(x)

    correlation_coefficient = np.corrcoef(y, predictions)[0, 1]
    correlation_description = get_correlation_description(correlation_coefficient)
    residuals = y-(model.predict(x))
    srd_resid = np.sum(residuals ** 2)


    stand_error = np.sqrt(srd_resid/ (len(y) - 2))
    
    output = {
        'predictions': predictions.tolist(),
        'stand_error': stand_error,
        'correlation_coefficient': correlation_coefficient,
        'correlation_description': correlation_description,
        'equation': {'coef': model.coef_.tolist(), 'intercept': model.intercept_, 'num_vars': len(independent_vars)},
        'independent_vars': independent_vars
    }

    plot_urls = plot_data(data, dependent_var, independent_vars, predictions, model)
    output['plot_urls'] = plot_urls


    return render_template('output-page.html', output=output)


def get_correlation_description(correlation_coefficient):
    if correlation_coefficient > 0.8:
        return "Very Strong Positive Correlation"
    elif correlation_coefficient > 0.6:
        return "Strong Positive Correlation"
    elif correlation_coefficient > 0.4:
        return "Moderate Positive Correlation"
    elif correlation_coefficient > 0.2:
        return "Weak Positive Correlation"
    elif correlation_coefficient >= -0.2:
        return "No Correlation"
    elif correlation_coefficient > -0.4:
        return "Weak Negative Correlation"
    elif correlation_coefficient > -0.6:
        return "Moderate Negative Correlation"
    elif correlation_coefficient > -0.8:
        return "Strong Negative Correlation"
    else:
        return "Very Strong Negative Correlation"

def plot_data(data, dependent_var, independent_vars, predictions, model):
    plot_urls = []
    
    for var in independent_vars:
        fig, ax = plt.subplots()
        ax.scatter(data[var], data[dependent_var], color='green', label='Data Points')
        
        # Calculate regression line
        regression_line = data[var] * model.coef_[independent_vars.index(var)] + model.intercept_
        
        ax.plot(data[var], regression_line, color='red', label='Best Fit Line')
        ax.set_xlabel(var)  
        ax.set_ylabel(dependent_var)
        ax.set_title(f'{dependent_var} vs {var}')
        ax.legend()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()

        plot_urls.append(plot_url)

    return plot_urls

if __name__ == '__main__':
    app.run(debug=True)
