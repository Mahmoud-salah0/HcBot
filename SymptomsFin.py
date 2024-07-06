from joblib import load
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import webbrowser



symptom_list = []

df = pd.read_excel(r'dataset.xlsx')

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

from fuzzywuzzy import process

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
        print("No matching diseases found")
        return False, "No matching diseases found"
    else:
        max_indices = similarity_scores.argmax(axis=0)
        symptoms = set()
        for i in max_indices:
            if similarity_scores[i] == max_score:
                # Split the symptoms and remove the first one (the disease name)
                symptoms_list = df.iloc[i]['Symptoms'].split(',')
                symptoms.update(set(symptoms_list[1:]))  # Exclude the first symptom
            # Output results
        print("The symptoms of", user_disease, "are:")
        for sym in symptoms:
            print(str(sym).capitalize())
    return True, symptoms


def user_conversation():
    symptom = input("Please enter one symptom\n")
    predicted = extract_symptom(symptom, list(symptoms.keys()))

    y = input(
        "Did you mean " + predicted.lower().replace("_", " ").strip() + "? (y/n)\n"
    )
    if y == "y":
        symptom_list.append(predicted)
        z = input("Do you want to add another symptom? (y/n)\n")
        if z == "y":
            user_conversation()
        elif z == "n":
            disease = predict_disease_from_symptom(symptom_list)
            print("The most likely disease based on your symptoms is: " + str(disease) + "\n")
            webSearch = input("Do you want some information about your predicted disease online? (y/n)\n")
            if(webSearch == "y"):
                url = "https://www.google.com/search?q=" + "Information about " + disease
                webbrowser.open(url)
    elif y == "n":
        user_conversation()


def chatbot():
    print(
        "Welcome to our Disease Prediction Chatbot!\nI'm here to help you understand the potential diseases you might be suffering from based on your symptoms.\n")
    name = input(
        "Can you firstly provide me with your name?\n"
    )
    print("Hello " + name + "! " + "\nWe hope we can be of help!\nDo you want to begin with your diagnosis?")
    input()
    print("Sounds great " + name + "! Let's begin, Note that the more symptoms you provide the model, the better the prediction will be")
    while (True):
        global symptom_list
        symptom_list.clear()
        user_conversation()
        tryAgain = input("We wish you a speedy recovery " + name + "! Do you want to predict again? (y/n)\n")
        if(tryAgain == "n"):
            break
    print("Hope we have provided some help " + name + ". Don't forget to ask for professional help!\nThanks for using HcBot!")




def mainChoices():
    x = input("choose:\n"
              "1-Enter your symptom\n"
              "2-Enter your disease to check symptoms\n"
              "3-Choose from a symptom list (More accurate)\n")
    if x == "1": user_conversation()
    if x == "2":
        disease = input("Enter One Disease to check symptoms\n")
        get_symptoms(disease)
    else:
        print("Invalid choice")


chatbot()

