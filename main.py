from flask import Flask,render_template,request
app=Flask(__name__)
import pickle

file=open('model1.pkl','rb')
Lr=pickle.load(file)
file.close()

@app.route('/about')
def predict1() :
	return render_template('about.html')

@app.route('/',methods=["GET","POST"])

def hello_world():
    if request.method=='POST':
       myDict=request.form
       
       fever=float(myDict['fever'])
      
       age=int(myDict['age'])

       pain=int(myDict['pain'])
       
       runnyNose=int(myDict['runnyNose'])

       diffBreath=int(myDict['diffBreath'])

       inF=[[fever,pain,age,runnyNose,diffBreath]]
 
       inf_prob=Lr.predict_proba(inF)[0][1]
       print(inf_prob)
       return  render_template('show.html', inp=round(inf_prob *100))

    return  render_template('index.html')
if __name__=='__main__':
    app.run(debug=True)