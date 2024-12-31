from flask import Flask, render_template, request
from src.Pipelines.prediction_pipeline import predict_pipeline, CustomData

app=Flask(__name__)

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predictdata', methods=["GET", "POST"])

def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    
    else:
        data=CustomData(
            sex=request.form.get('sex'),
            bmi=float(request.form.get('bmi')),
            children=int(request.form.get('children')),
            smoker=request.form.get('smoker'),
            region=request.form.get('region'),
            
        )
        pred_df=data.get_data_as_a_fram()
        print(pred_df)
        
        predictpipeline=predict_pipeline()
        result=predictpipeline.predict(pred_df)
        
        return render_template("result.html", final_result=result[0])
    
    
if __name__=='__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
    