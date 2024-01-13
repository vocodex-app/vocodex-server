from analyze_lyrics import get_stem
from database import  get_song_with_emotion

def fetch_music_based_on_emotion(emotion):
    return get_song_with_emotion(emotion)

"""
def fetch_music_based_on_activity(text):
    result = []
    s = text.split()
    s = get_stem(s)
    for emotion, song_list in data.items():
        # print(song_list)
        for name, song_info in song_list.items():
            # print(name)
            # print(song_info)
            song_info = data[emotion][name]
            keywords = song_info['keywords']
            # print(song_info)
            # print(keywords)
            for word in s:
                if word in keywords:
                    print(name)
                    result.append(name)
    return result

"""
if __name__ == "__main__":
    print(fetch_music_based_on_emotion('joy'))
    #result = fetch_music_based_on_activity('playing')
    #print(result)
