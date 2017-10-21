from flask import request, render_template, Flask, redirect, url_for
from werkzeug.utils import secure_filename
import dataset
import os
import requests
from ast import literal_eval
import smtplib
import time
import _thread as thread
from wrapper import optimize

# from UL.wrapper import optimize

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index_std.html")

@app.route("/constraints/hp_input")
def hp_input():
    return render_template("/constraints/hp_input.html")

@app.route("/constraints/RidgeR_Lasso")
def Ridge_Regression():
    return render_template("/constraints/RidgeR_Lasso.html")

@app.route("/constraints/Logic")
def Logic():
    return render_template("/constraints/Logic.html")

@app.route("/constraints/SVR_C")
def SVR_C():
    return render_template("/constraints/SVR_C.html")

@app.route("/eval")
def eval():
    return render_template("/eval.html")

@app.route("/index_std")
def index_std():
    return render_template("index_std.html")

@app.route("/index_cus")
def index_cus():
    return render_template("index_cus.html")

@app.route("/response_std", methods=['POST'])
def action_std():
    learn_type = request.form["type"]
    algo = request.form["algo"]
    model = request.form["model"]

    if(learn_type=="Supervised"):
        if(model=="Lasso" or model=="Ridge Regression"):
            alpha_min = request.form["alpha_min"]
            alpha_max = request.form["alpha_max"]
            eval_type = request.form["eval"]  
            #print(alpha_min,alpha_max,algo, model,learn_type)

        if(model=="Logistic Regression"):
            c_min = request.form["c_min"]
            c_max = request.form["c_max"]
            eval_type = request.form["eval"]  
            #print(c_min,c_max,algo, model,learn_type)

        if(model=="SVR" or model=="SVC"):
            c_min = request.form["c_min"]
            c_max = request.form["c_max"]
            gamma_min = request.form["gamma_min"]
            gamma_max = request.form["gamma_max"]
            eval_type = request.form["eval"]  
            #print(gamma_min,gamma_max,c_min,c_max,algo, model,learn_type)

    if(learn_type=="Unsupervised"):
        if(model=="Bandit" or model=="Cart Pole"):
            lr_min = request.form["lr_min"]
            lr_max = request.form["lr_max"]
            rd_min = request.form["rd_min"]
            rd_max = request.form["rd_max"]
            mt = request.form["mt"]
            n_iter = request.form["n_iter"]
            n_epis = request.form["n_epis"]
            if(algo == "Bayesian"):
                if(model == "Bandit"):
                    optimize("Bayesian", "Bandit", meta_trials=int(mt), n_iter=int(n_iter), lr=[float(lr_min), float(lr_max)], n_episodes=int(n_epis), bandits=[0.5, 0, -0.3, -0.6], e=0.2)
                else:
                    optimize("Bayesian", "Bandit", meta_trials=int(mt), n_iter=int(n_iter), lr=[float(lr_min), float(lr_max)], n_episodes=int(n_epis))
            else:
                optimize("Optunity", None, f=opt_cartpole, num_evals=int(n_iter), learn_rate=[float(lr_min), float(lr_max)], rew_decay=[float(rd_min), float(rd_max)])

    time.sleep(2)
    return redirect("https://dh5.aiam-dh.com/melbootcamp02/en-US/app/aaam_hyperparameter_optimization/dashboards")


@app.route("/response_cus", methods=['POST'])
def action_cus():
    time.sleep(2)
    return redirect("http://www.google.com")

app.run(host="0.0.0.0", debug=True, port=8081)

