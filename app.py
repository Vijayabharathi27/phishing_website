from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load('rf_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # Render your HTML form

@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    
    # Transform the input URL using the vectorizer
    url_vectorized = vectorizer.transform([url])
    
    # Make the prediction
    prediction = model.predict(url_vectorized)[0]
    
    # Return the prediction result
    return jsonify({'result': prediction})

if __name__ == '__main__':
    app.run(debug=True)
