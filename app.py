import re
from flask import Flask, render_template, flash, redirect, url_for, session, logging, request, jsonify
import pickle
import random
import secrets
import webbrowser as wbbb
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
import numpy as np
from fuzzywuzzy import process

app=Flask(__name__)

#Load the model
model = pickle.load(open("newModel.pkl","rb"))

@app.route("/")
def home():
    return render_template('chatbot.html')

@app.route('/bmi',methods=['GET','POST'])
def bmi():
    return render_template('bmi.html')

@app.route('/fnh',methods=['GET','POST'])
def fnh():
    return render_template('FNH.html')
@app.route('/about',methods=['GET','POST'])
def cnt():
    return render_template('About.html')

userSession = {}
all_result = {
    'name':'',
    'age':0,
    'gender':'',
    'symptoms':[]
}
def getDiseaseInfo(keywords):
    results = wbbb(keywords, region='wt-wt', safesearch='Off', time='y')
    return results[0]['body']

def make_token(length=16):
    """
    Creates a cryptographically-secure, URL-safe string
    """
    return secrets.token_urlsafe(length)

def index_auth(userSession):
    """
    Generate a session ID and initialize it with a value of -1 in userSession dictionary
    """
    session_id = make_token()
    userSession[session_id] = 0
    return session_id



def predict_symptom(user_input, symptom_list):
    # Convert user input to lowercase and split into tokens
    user_input_tokens = user_input.lower().replace("_"," ").split()

    # Calculate cosine similarity between user input and each symptom
    similarity_scores = []
    for symptom in symptom_list:
        # Convert symptom to lowercase and split into tokens
        symptom_tokens = symptom.lower().replace("_"," ").split()

        # Create count vectors for user input and symptom
        count_vector = np.zeros((2, len(set(user_input_tokens + symptom_tokens))))
        for i, token in enumerate(set(user_input_tokens + symptom_tokens)):
            count_vector[0][i] = user_input_tokens.count(token)
            count_vector[1][i] = symptom_tokens.count(token)

        # Calculate cosine similarity between count vectors
        similarity = cosine_similarity(count_vector)[0][1]
        similarity_scores.append(similarity)

    # Return symptom with highest similarity score
    max_score_index = np.argmax(similarity_scores)
    return symptom_list[max_score_index]



# Load the dataset into a pandas dataframe
df = pd.read_excel(r'dataset.xlsx')

# Get all unique symptoms
symptoms = set()
for s in df['Symptoms']:
    for symptom in s.split(','):
        symptoms.add(symptom.strip())


symptoms = {'itching': 0, 'skin_rash': 0, 'nodal_skin_eruptions': 0, 'continuous_sneezing': 0,
            'shivering': 0, 'chills': 0, 'joint_pain': 0, 'stomach_pain': 0, 'acidity': 0, 'ulcers_on_tongue': 0,
            'muscle_wasting': 0, 'vomiting': 0, 'burning_micturition': 0, 'spotting_ urination': 0, 'fatigue': 0,
            'weight_gain': 0, 'anxiety': 0, 'cold_hands_and_feets': 0, 'mood_swings': 0, 'weight_loss': 0,
            'restlessness': 0, 'lethargy': 0, 'patches_in_throat': 0, 'irregular_sugar_level': 0, 'cough': 0,
            'high_fever': 0, 'sunken_eyes': 0, 'breathlessness': 0, 'sweating': 0, 'dehydration': 0,
            'indigestion': 0, 'headache': 0, 'yellowish_skin': 0, 'dark_urine': 0, 'nausea': 0, 'loss_of_appetite': 0,
            'pain_behind_the_eyes': 0, 'back_pain': 0, 'constipation': 0, 'abdominal_pain': 0, 'diarrhoea': 0,
            'mild_fever': 0,
            'yellow_urine': 0, 'yellowing_of_eyes': 0, 'acute_liver_failure': 0, 'fluid_overload': 0,
            'swelling_of_stomach': 0,
            'swelled_lymph_nodes': 0, 'malaise': 0, 'blurred_and_distorted_vision': 0, 'phlegm': 0,
            'throat_irritation': 0,
            'redness_of_eyes': 0, 'sinus_pressure': 0, 'runny_nose': 0, 'congestion': 0, 'chest_pain': 0,
            'weakness_in_limbs': 0,
            'fast_heart_rate': 0, 'pain_during_bowel_movements': 0, 'pain_in_anal_region': 0, 'bloody_stool': 0,
            'irritation_in_anus': 0, 'neck_pain': 0, 'dizziness': 0, 'cramps': 0, 'bruising': 0, 'obesity': 0,
            'swollen_legs': 0,
            'swollen_blood_vessels': 0, 'puffy_face_and_eyes': 0, 'enlarged_thyroid': 0, 'brittle_nails': 0,
            'swollen_extremeties': 0,
            'excessive_hunger': 0, 'extra_marital_contacts': 0, 'drying_and_tingling_lips': 0, 'slurred_speech': 0,
            'knee_pain': 0, 'hip_joint_pain': 0, 'muscle_weakness': 0, 'stiff_neck': 0, 'swelling_joints': 0,
            'movement_stiffness': 0,
            'spinning_movements': 0, 'loss_of_balance': 0, 'unsteadiness': 0, 'weakness_of_one_body_side': 0,
            'loss_of_smell': 0,
            'bladder_discomfort': 0, 'foul_smell_of_urine': 0, 'continuous_feel_of_urine': 0, 'passage_of_gases': 0,
            'internal_itching': 0,
            'toxic_look_(typhos)': 0, 'depression': 0, 'irritability': 0, 'muscle_pain': 0, 'altered_sensorium': 0,
            'red_spots_over_body': 0, 'belly_pain': 0, 'abnormal_menstruation': 0, 'dischromic _patches': 0,
            'watering_from_eyes': 0,
            'increased_appetite': 0, 'polyuria': 0, 'family_history': 0, 'mucoid_sputum': 0, 'rusty_sputum': 0,
            'lack_of_concentration': 0,
            'visual_disturbances': 0, 'receiving_blood_transfusion': 0, 'receiving_unsterile_injections': 0, 'coma': 0,
            'stomach_bleeding': 0, 'distention_of_abdomen': 0, 'history_of_alcohol_consumption': 0,
            'fluid_overload.1': 0,
            'blood_in_sputum': 0, 'prominent_veins_on_calf': 0, 'palpitations': 0, 'painful_walking': 0,
            'pus_filled_pimples': 0,
            'blackheads': 0, 'scurring': 0, 'skin_peeling': 0, 'silver_like_dusting': 0, 'small_dents_in_nails': 0,
            'inflammatory_nails': 0,
            'blister': 0, 'red_sore_around_nose': 0, 'yellow_crust_ooze': 0, 'brain_hemorrhage': 0}


def extract_symptom(user_input, symptom_list):
    # Use fuzzy matching to find the closest symptom to the user input
    closest_match, similarity_score = process.extractOne(user_input, symptom_list)
    return closest_match




def predict_disease_from_symptom(symptom_list):
    model = load(r'model.joblib')
    vectorizer = load(r'vectorizer.joblib')

    # Vectorize the user's symptoms
    user_symptoms = ', '.join(symptom_list)
    user_X = vectorizer.transform([user_symptoms])

    predicted_disease = model.predict(user_X)

    return predicted_disease[0]


def get_symptoms(user_disease):
    # Vectorize diseases using CountVectorizer
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Disease'])
    user_X = vectorizer.transform([user_disease])

    # Compute cosine similarity between user disease and dataset diseases
    similarity_scores = cosine_similarity(X, user_X)

    # Find the most similar disease(s)
    max_score = similarity_scores.max()
    if max_score < 0.7:
        
        return False, "No matching diseases found"
    else:
        max_indices = similarity_scores.argmax(axis=0)
        symptoms = set()
        for i in max_indices:
            if similarity_scores[i] == max_score:
                # Split the symptoms and remove the first one (the disease name)
                symptoms_list = df.iloc[i]['Symptoms'].split(',')
                symptoms.update(set(symptoms_list[1:]))  # Exclude the first symptom
    return True, symptoms

        
symptom_list = []

@app.route('/process', methods=['GET', 'POST'])
def chat_msg():
    global predicted  
    global userSession
    
    # Get the JSON data from the request
    req_data = request.get_json()

    # Extract message and sessionId from the JSON data
    user_message = req_data.get("message", "").lower()
    sessionId = req_data.get("sessionId", "")
    
   
     # Retrieve current state from userSession
    currentState = userSession.get(sessionId, 0)
    
    response = ""
   
    if currentState == 0:
        response = "Hi,What is your name?"
        userSession[sessionId] = 1

    elif currentState == 1:
        # Handling user's name input
        global username
        username = user_message
        response = f"Hi {username}, to predict your disease based on symptoms, we need some information about you. Please provide your age."
        all_result['name'] = username
        userSession[sessionId] = 2

    elif currentState == 2:
        # Handling user's age input
        pattern = r'\d+'
        result = re.findall(pattern, user_message)
        if len(result) == 0:
            response = "Invalid input. Please provide a valid age."
        else:
            age = float(result[0])
            if age <= 0 or age >= 130:
                response = "Invalid input. Please provide a valid age."
            else:
                all_result['age'] = age
                response = "Do you want to \n 1-Predict disease \n or \n 2-Get symptoms"
                userSession[sessionId]=3

    elif currentState== 3:
        if user_message=="1" or user_message.lower()=="predict disease":
            response = "please add a symptom"
            userSession[sessionId] = 10
        elif user_message=="2" or user_message.lower()=="get symptoms":
            response="please provide me with the disease"
            userSession[sessionId]=8
        

        

    elif currentState == 10:
        # Handling user's symptom input
        global predicted 
        predicted = None
        predicted = extract_symptom(user_message, list(symptoms.keys()))
        response = "Did you mean " + predicted.lower().replace("_", " ").strip() + "? (yes/no)"
        userSession[sessionId] = 4

    elif currentState == 4:
        
        if user_message.lower() == 'yes':
            # User confirmed the predicted symptom
            symptom_list.append(predicted)
            response = "Symptom added. Do you want to add another symptom? (yes/no)"
            userSession[sessionId] = 5
        elif user_message.lower() == 'no':
            # User didn't confirm the predicted symptom, prompt for the correct one
            response = "Please enter the correct symptom."
            userSession[sessionId] = 10
        else:
            # Invalid input, prompt again
            response = "Invalid input. Please enter 'yes' or 'no'."

    elif currentState == 5:
        # Handling user's decision to add another symptom
        if user_message.lower() == 'yes':
            # User wants to add another symptom
            response = "Please enter another symptom"
            userSession[sessionId] = 10
        elif user_message.lower() == 'no':
            
            disease = predict_disease_from_symptom(symptom_list)
            response = f"The most likely disease based on your symptoms is: <span style='color: red;'>{disease}</span>\n"

            response += "Do you want some information about your predicted disease online? (yes/no)\n"
            userSession[sessionId] = 6
           
           
        else:
            # Invalid input, prompt again
            response = "Invalid input. Please enter 'yes' or 'no'."
    elif currentState == 6:
        
        if user_message.lower() == 'yes':
            # User wants information about the predicted disease online
            disease = predict_disease_from_symptom(symptom_list)
            if(user_message == "yes"):
                url = "https://www.google.com/search?q=" + "Information about " + disease
                wbbb.open(url)
          
            response = f"We wish you a speedy recovery {username}! Do you want to predict again? (yes/no)"
            
            userSession[sessionId] = 7
        elif user_message.lower() == 'no':
            # User doesn't want information about the predicted disease online
            response = f"We wish you a speedy recovery {username}! Do you want to predict again? (yes/no)"
            userSession[sessionId] = 7
        else:
            # Invalid input, prompt again
            response = "Invalid input. Please enter 'yes' or 'no'."

    elif currentState == 7:
         if user_message.lower() == 'yes':
                response = "Please enter one symptom"
                userSession[sessionId] = 10
            
         elif user_message.lower() == 'no':
            
             response = f"Thanks for using HcBot,{username}!"
             """userSession[sessionId] = 2"""
         else:
            # Invalid input, prompt again
            response = "Invalid input. Please enter 'yes' or 'no'."

    elif currentState ==8:
       list_of_symptoms = set()
       x,list_of_symptoms=get_symptoms(user_message)
       response = f"The symptoms based on your disease are: <span style='color: green;'>{', '.join(list_of_symptoms)}</span>\n"
       userSession[sessionId]=9

    elif currentState == 9:
        
        if user_message.lower() == "yes":
            response = "please mention the disease name"
            userSession[sessionId]=8

        elif user_message.lower() =="no":
            response = f"Thanks for using HcBot,{username}!"
    

        # Generate HTML response
    html_response = generate_html_response(response)
    if currentState == 1:
            html_response += generate_html_response("How old are you?", div_class="message sender")
    elif currentState == 8:
            html_response += generate_html_response("Do you want to get symptoms for another disease? (yes/no)", div_class="message sender")
  
    return jsonify({'status': 'OK', 'html_response': html_response})


def generate_html_response(message, div_class="message sender"):
    """
    Generates HTML response for the given message
    """
    # Create a separate <div> for each message with appropriate class and margin
    return f'<div class="{div_class}" style="margin-bottom: 10px;"><div class="message-bubble">{message}</div></div>'


if __name__=="__main__":
    app.run(debug=False)
