import firebase_admin
from firebase_admin import credentials, firestore
import random

cred = credentials.Certificate("vocodex-72355-firebase-adminsdk-s60ug-abc1fa30aa.json")
firebase_admin.initialize_app(cred)

db = firestore.client()  # fs connection
collection = 'musicRecommendation'


def get_song_with_emotion(emotion) -> str:
    # fetch song which has field with given emotion
    docs = db.collection(collection).where(u'emotion', u'==', emotion).stream()
    # get song as json
    song_id = random.choice([doc.id for doc in docs])
    song = db.collection(collection).document(song_id).get().to_dict()
    return song


"""
def get_data(emotion):
    doc_ref = db.collection(collection).document(emotion)
    doc = doc_ref.get()
    if doc.exists:
        return doc.to_dict()
    else:
        return {}


def get_all_data():
    data = {}
    docs = db.collection(collection).stream()

    for doc in docs:
        data[doc.id] = doc.to_dict()
    return data


def update_data(emotion, data):
    doc_ref = db.collection(collection).document(emotion)
    doc_ref.set(data, merge=True)
"""