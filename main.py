from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import os
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

import discord
import asyncio
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer

data = pd.read_csv("samples.csv")
texts = []
for i, label in enumerate(data['Category']):
    texts.append(data['Message'][i])

texts = np.asarray(texts)

client = discord.Client()

@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    input = tokenizer.texts_to_sequences([message.content])[0]
    spam_words = loaded_model.predict(input)
    spam_percent = 0
    for spam in spam_words:
        spam_percent += spam[0]
    spam_percent /= len(spam_words)
    await message.channel.send(str(spam_percent*100)[:6]+"\% spam")
    if spam_percent >0.25:
        await message.author.kick()
        invite = await message.channel.create_invite(max_uses=1)
        await message.author.send(invite.url)

client.run('TOKEN')