from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = None

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model

    if model is None:
        model = joblib.load('model.pkl')

    # Obtenha os dados de entrada do formulário
    feature1 = float(request.form['feature1'])
    feature2 = float(request.form['feature2'])
    # Adicione mais campos conforme necessário para o seu modelo

    # Faça a previsão usando o modelo
    input_data = [[feature1, feature2]]  # Organize os dados em uma lista ou matriz adequada para o seu modelo
    prediction = model.predict(input_data)

    # Renderize a página de resultados com o resultado da previsão
    return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
