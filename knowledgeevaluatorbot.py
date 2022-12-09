import telebot
from telebot import types

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

bot = telebot.TeleBot('token')

#сами данные
df = pd.read_excel(r'C:\Users\krams\Desktop\project\данные_на_обучение.xlsx')
df = df.drop('Unnamed: 0', axis=1)
X = df.drop('percent', axis=1)
y = df['percent']

#модель
model = ElasticNet()
grid = {'alpha':[0.0001, 0.001, 0.01, 0.1, 1, 10, 100], 'l1_ratio':np.arange(0.0, 1.0, 0.1)}
grid_model = GridSearchCV(estimator=model, param_grid=grid, scoring='neg_mean_squared_error', cv=5, verbose=2)
grid_model.fit(X.values ,y)


@bot.message_handler(commands=['start'])
def welcome(message):
    #клава
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    item1 = types.KeyboardButton('Отправить массив данных')
    item2 = types.KeyboardButton("🪨Пример входных данных🪨")
    markup.add(item1, item2)

    # приветствие
    mess = f'Привет, <b>{message.from_user.first_name} {message.from_user.last_name}</b>, я бот который может проанализировать и предсказать твои оценки в будущем'
    bot.send_message(message.chat.id, mess, parse_mode='html', reply_markup=markup)

@bot.message_handler(content_types=['text'])
def conversation(message):
    if message.text == "🪨Пример входных данных🪨":
        photo = open('пример.jpg', 'rb')
        bot.send_photo(message.chat.id, photo)
        bot.send_message(message.chat.id, '8554544445445 - вот пример ваших данных')

    if message.text == 'Отправить массив данных':
        msg = bot.send_message(message.chat.id, 'Скидывайте ваш массив данных')
        bot.register_next_step_handler(msg, predict)

def predict(data):
    if str(data.text)[1] == '1':
        mess = "Я не умею определять оценки в 12 классе и выше, вы скинули неправильные данные"
        bot.send_message(data.chat.id, mess)
    arr = np.empty((0))
    try:

        for i in str(data.text):
            if str(data.text)[0] == '1':
                 number = i + str(data.text)[1]
                 data.text = str(data.text)[2]
                 arr = np.append(arr, int(number))
            else:
                 arr = np.append(arr, int(i))


        else:
            if arr.size == 14:
                arr = np.delete(arr, 1)
            y_pred = grid_model.predict(arr.reshape(1, -1))
            mess = f"Вот процент твоих возможных пятерок в 11 классе на основе твоих данных: {round(y_pred[0], 5)} %"
            bot.send_message(data.chat.id, mess)
            # пять символов для округления после запятой потому что выглядит круто


    except:
        mess = "У вас ошибка в ваших данных, проверте и отошлите еще раз"
        bot.send_message(data.chat.id, mess)
    print(arr)

bot.polling(none_stop=True)
