from flask import Flask, request, jsonify
import pickle
import pandas as pd

# init
app = Flask(__name__)

# open model
def open_model(model_path):
    """
    helper function for loading model
    """
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    return model

model_salary = open_model("pipeline_salary1.pkl")

def salary_inference(data, model=model_salary):
    """
    input : list with length 8 --> ['work_year', 'experience_level', 'employment_type', 'job_title','employee_residence', 'remote_ratio', 'company_location','company_size']
    output : predicted class : (idx, label)
    """
    columns = ['work_year', 'experience_level', 'employment_type', 'job_title','employee_residence', 'remote_ratio', 'company_location','company_size']
    data = pd.DataFrame([data], columns=columns)
    res = model.predict(data)
    return res[0]

@app.route('/')
def home():
    return "<h1>It Works!</h1>"

@app.route("/salary")
def salary_predict():
    args = request.args
    prediction_year = args.get("work_year", type=int, default=2022)
    experience = args.get("experience_level", type=object, default='MI')
    employment_type = args.get("employment_type", type=object, default='FT')
    job_title = args.get("job_title", type=object, default='Data Scientist')
    employee_residence = args.get("employee_residence", type=float, default='US')
    remote_ratio = args.get("remote_ratio", type=int, default=100)
    company_location = args.get("company_location", type=object, default='US')
    company_size = args.get("company_size", type=object, default='M')
    new_data = [prediction_year, experience, employment_type, job_title,
                employee_residence, remote_ratio,company_location,company_size]
    idx= salary_inference(new_data)
    response = jsonify(result=(round(float(idx),2)))
    return response

#@app.route("/salary",methods=['POST'])
#def salary_predict():
#    args = request.json
#    prediction_year = args.get("work_year")
#    experience = args.get("experience_level")
#    employment_type = args.get("employment_type")
#    job_title = args.get("job_title")
#    employee_residence = args.get("employee_residence")
#    remote_ratio = args.get("remote_ratio")
#    company_location = args.get("company_location")
#    company_size = args.get("company_size")
#    new_data = [prediction_year, experience, employment_type, job_title,
#                employee_residence, remote_ratio,company_location,company_size]
#    idx= salary_inference(new_data)
#    response = jsonify(result=round(float(idx),2))
#    return response

app.run(debug=True)