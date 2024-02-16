from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

def prediction(lst):
    filename = 'model/vehicleDataModel.pickle'
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    pred_value = model.predict([lst])
    return pred_value

@app.route('/data')
def get_time():
    return jsonify({
        'Name': "geek", 
        'Age': "22",
        'Date': "2023-03-04", 
        'programming': "python"
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    ram = int(data['ram'])
    weight = float(data['weight'])
    company = data['company']
    typename = data['typename']
    opsys = data['opsys']
    cpu = data['cpuname']
    gpu = data['gpuname']
    touchscreen = data['touchscreen']
    ips = data['ips']

    feature_list = []

    feature_list.append(int(ram))
    feature_list.append(float(weight))
    feature_list.append(1 if touchscreen else 0)
    feature_list.append(1 if ips else 0)

    company_list = ['acer','apple','asus','dell','hp','lenovo','msi','other','toshiba']
    typename_list = ['2in1convertible','gaming','netbook','notebook','ultrabook','workstation']
    opsys_list = ['linux','mac','other','windows']
    cpu_list = ['amd','intelcorei3','intelcorei5','intelcorei7','other']
    gpu_list = ['amd','intel','nvidia']

    def traverse_list(lst, value):
        for item in lst:
            if item == value:
                feature_list.append(1)
            else:
                feature_list.append(0)

    traverse_list(company_list, company)
    traverse_list(typename_list, typename)
    traverse_list(opsys_list, opsys)
    traverse_list(cpu_list, cpu)
    traverse_list(gpu_list, gpu)

    pred_value = prediction(feature_list)
    pred_value = np.round(pred_value[0], 2) * 221

    return jsonify({'prediction': pred_value})

if __name__ == '__main__':
    app.run(debug=True)
