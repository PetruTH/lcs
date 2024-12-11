from flask import Flask, render_template, request
import transformers
import spacy

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

# Emotion labels
labels = {
    "None": "",
    "LABEL_0": "sadness",
    "LABEL_1": "joy",
    "LABEL_2": "love",
    "LABEL_3": "anger",
    "LABEL_4": "fear",
    "LABEL_5": "surprise",
}

def predict_emotion(phrase, emotion_pipeline):
    """
    Predict the emotion of the input phrase using a pre-trained model.
    
    Args:
        phrase (str): Input text.
        
    Returns:
        tuple: The predicted emotion label and its confidence score.
    """
    predictions = emotion_pipeline(phrase)[0]  # Get the scores for all labels
    # Find the label with the highest score
    top_prediction = max(predictions, key=lambda x: x['score'])
    return top_prediction['label'], top_prediction['score']

def analyze_phrase(phrase):
    """
    Analyze the input phrase using SpaCy and return details about each word.
    
    Args:
        phrase (str): Input text.
        
    Returns:
        list: List of word details (POS, DEP).
    """
    doc = nlp(phrase)
    word_details = [
        {"text": token.text, "pos": token.pos_, "dep": token.dep_}
        for token in doc
    ]
    return word_details

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    emotion_pipeline = transformers.pipeline(
        "text-classification",
        model="./emotion_model",
        tokenizer="./emotion_model",
        return_all_scores=True
    )

    predicted_label = "None"
    confidence_score = "None"
    phrase = ''
    word_details = []
    
    if request.method == 'POST':
        phrase = request.form.get('phrase', '')
        if phrase:
            # Predict emotion
            predicted_label, confidence_score = predict_emotion(phrase, emotion_pipeline)
            # Analyze phrase with SpaCy
            word_details = analyze_phrase(phrase)

    print(word_details)

    return render_template(
        'index.html',
        phrase=phrase,
        predicted_label=labels[predicted_label],
        confidence_score=confidence_score,
        word_details=word_details
    )

if __name__ == '__main__':
    app.run(debug=True)


# TODO: add 2 new LLMs and compare them
# TODO: improve hover info