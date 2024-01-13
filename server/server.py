from flask import Flask, request, jsonify

from music_recommendation import  fetch_music_based_on_emotion
from textEmotion import predict_text_emotion

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return "Music Recommendation"


@app.route('/get_emo', methods=['GET','POST'])
def get_emo():
    content = request.json
    context = content['text']
    emotion = predict_text_emotion(text=context)
    return jsonify(emotion)

@app.route('/fetch_music_based_on_text/', methods=['GET', 'POST'])
def get_music_based_on_keywords():
    content = request.json
    context = content['text']
    emotion = predict_text_emotion(text=context)
    data = fetch_music_based_on_emotion(emotion)

    return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
