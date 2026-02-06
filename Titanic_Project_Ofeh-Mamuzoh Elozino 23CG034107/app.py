from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model/titanic_survival_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Getting data from the HTML form
    # Note: Ensure these names match the 'name' attribute in your HTML input tags
    try:
        pclass = int(request.form['pclass'])
        age = float(request.form['age'])
        sibsp = int(request.form['sibsp'])
        parch = int(request.form['parch'])
        fare = float(request.form['fare'])
        gender = 1 if request.form['gender'] == 'male' else 0 # Simple encoding

        # Arrange features in a 2D array for the model
        features = np.array([[pclass, gender, age, sibsp, parch, fare]])
        prediction = model.predict(features)
        
        result = "Survived" if prediction[0] == 1 else "Did Not Survive"
        
        return render_template('index.html', prediction_text=f'Result: {result}')
    
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
