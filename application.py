from flask import Flask,request,render_template,jsonify
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


application=Flask(__name__)

app=application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])

def predict_datapoint():
    if request.method=='GET':
        return render_template('index.html')
    
    else:
        data=CustomData(
            Pclass=float(request.form.get('Pclass')),
            Sex = request.form.get('Sex'),
            Age = float(request.form.get('Age')),
            SibSp = float(request.form.get('SibSp')),
            Parch = float(request.form.get('Parch')),
            Fare = float(request.form.get('Fare'))            
        )
       
        final_new_data=data.get_data_as_dataframe()
        predict_pipeline=PredictPipeline()
        pred=predict_pipeline.predict(final_new_data)

        results=round(pred[0],2)

        return render_template('results.html',final_result=results)


if __name__=="__main__":
    app.run(host='0.0.0.0',port=5005, debug=True)
    


