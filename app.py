print("STARTING APP...")

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

try:
    model = pickle.load(open('model/toxic_model.pkl', 'rb'))
    vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))
    print("Model loaded successfully ✅")
except Exception as e:
    print("ERROR LOADING MODEL ❌:", e)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['comment']
    data = vectorizer.transform([text])
    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "Toxic 😡"
    else:
        result = "Not Toxic 😊"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)