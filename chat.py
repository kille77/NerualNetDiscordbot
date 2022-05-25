#chat
import enum
import json
import random
import pickle
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import discord
import os


#sanojen perusmuodon muokkaus ja json-tiedosto käyttöön, sanat ja luokat ja neuralNW-malli
lemmatizer=WordNetLemmatizer()
intents=json.loads(open('intents.json').read())
words=pickle.load(open('words.pkl', 'rb'))
classes=pickle.load(open('classes.pkl', 'rb'))
model=load_model('OmaCHatti_model.h5')
#funktiot joilla vastaus poimitaan
def bag_of_words(lause):
    #tekee listan lauseen eri sanoista, jotta voidaan sanojen perusteella laskea todennäköisyyksiä
    lauseen_sanat = nltk.word_tokenize(lause)
    lauseen_sanat=[lemmatizer.lemmatize(word) for word in lauseen_sanat]
    bag=[0] * len(words)
    for w in lauseen_sanat:
        for i, word in enumerate(words):
            if word ==w:
                bag[i]=1
    return np.array(bag)

def predict_class(lause):
     # hakee tilastollisen todennäköisyyden perusteella todennäköisimmän vastauksen
    bow=bag_of_words(lause)
    res=model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    results=[[i,r] for i, r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list=[]
    for r in results:
        return_list.append({'intent':classes[r[0]],'probability':str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
     #hakee vastauksen tilastollisen ennakoinnin perusteella
     #predict_class - funktion poimiman korkeimman todennäköisyyden perusteella
    tag=intents_list[0]['intent']
    list_of_intents=intents_json['intents']
    for i in list_of_intents:
        if i['tag']==tag:
            result=random.choice(i['responses'])
            break
    return result



#pääohjelma

client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
        if message.author == client.user:
            return
    
        kysymys=message.content
        ints=predict_class(kysymys)
        vastaus=get_response(ints,intents)
        await message.channel.send(vastaus)
        

client.run('BOT:KEY:HERE')


