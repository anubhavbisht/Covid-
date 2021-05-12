import pickle

from flask import Flask, render_template, request
app = Flask(__name__)

# open a file where you store the pickled data
file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    if request.method == "POST":
        from twiliocred import my_sid, my_auth_token, my_cell, my_twilio
        from twilio.rest import Client
        mydict = request.form
        name = str(mydict['name'])
        address = str(mydict['address'])
        contact = str(mydict['no.'])
        fever = int(mydict['fever'])
        age = int(mydict['age'])
        bodypain = int(mydict['pain'])
        runnynose = int(mydict['runny nose'])
        diffbreath = int(mydict['difficulty'])
        # code for interfernce
        inputfeatures = [fever, bodypain, age, runnynose, diffbreath]
        infectionprobabilty = clf.predict_proba([inputfeatures])[0][1]
        prob = round(infectionprobabilty*100)
        # print(infectionprobabilty)
        client = Client(my_sid, my_auth_token)
        my_msg = "hello {} has completed the test,and his/her infection probability rate is {}%, and here are their contacts{} {}".format(
            name, str(prob), contact, address)  # enter your message
        message = client.messages.create(
            to=my_cell, from_=my_twilio, body=my_msg)
        return render_template('show.html', inf=round(infectionprobabilty*100))
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
