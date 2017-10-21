from flask import request, render_template, Flask
import os
import _thread as thread

app = Flask(__name__)



'''
Displays the user input page at "{URL}/index" and posts user entered data at itself.
'''

@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get post data and displays the submitted page
        fieldID1 = request.form["fieldID1"]
        fieldID2 = request.form["fieldID2"]
        print(fieldID1, fieldID2)
        return render_template("SubmitSuccessPage.html")
    else:
        # Displays the user input page (index page)
        return render_template("index.html")


@app.errorhandler(404)
def Catch(e=None):
    return render_template("404.html")



app.run(host="0.0.0.0", debug=False, port=80)
