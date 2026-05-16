"""
End-2-end генерация рифмованного четверостишья с помощью отфайнтюненной GPT с маркировкой ударений.
Используется вводимая затравка в виде словосочетания (именная группа).

09-12-2021 Подключен StressedGptTokenizer и используется tokenizer_config.json
09-12-2021 Доработка для телеграм-бота
11-12-2021 Переписываем код top_t+top_p сэмплинга, чтобы банить цепочки с повтором рифмуемого слова.
14-12-2021 Добавлен код для автоматической пакетной оценки качества генерации.
18-12-2021 Добавлена коррекция пробелов после декодера, модуль whitespace_normalization
27-01-2022 Эксперимент с управлением форматом генерации с помощью тегов [стих | одностишье | двустишье | порошок]
08.02.2022 В ранжирование результатов генерации добавлена проверка бедной рифмовки (включая повторение рифмуемого слова)
21.02.2022 Добавлена модель генерации следующих 4х строчек по первому четверостишью.
26.02.2022 Рефакторинг - генерация стиха, ранжировка и прочее вынесены в отдельный модуль, а тут остается только фронт
03.05.2022 Добавлена кнопка "Новая тема" для формирования новых саджестов
08.05.2022 Эксперимент с генерацией рубаи
08.05.2022 Из телеграм-версии исключен режим моностихов
24.05.2022 Добавлен второй уровень рубрикации четверостиший - выделены рубаи, частушки, мистика и т.д.
27.05.2022 В режиме генерации рубаи не делается продолжение (след. 4 строки по первым 4м)
22.10.2022 Реализован повтор попыток генерации с повышенной температурой, если все сгенерированные варианты отсеяны фильтрами качества
23.10.2022 Эксперимент с отдельной моделью для генерации длинных стихов: LongPoetryGenerator
27.10.2022 Если генерация в модели long poems ничего не дала, запускаем генерацию в старой модели 4-строчников.
11.11.2022 Большая чистка: убираем все жанры генерации, кроме лирики, меняем стартовый экран. Новое ядро генератора.
21.11.2022 Переход на модель GPT 355, датасет файнтюна без жанров вообще
01.03.2023 Введена трансляция пользовательских затравок
07.03.2023 Добавлен вывод расширенной статистики по посетителям и оценкам по команде /stat
"""
import collections
import datetime
import os
import logging
import argparse
import traceback
import getpass
import itertools
import re

import telegram
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler
from telegram import ReplyKeyboardMarkup, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardRemove, Update

import tensorflow as tf
import torch

from generative_poetry.init_logging import init_logging
from generative_poetry.poetry_seeds import SeedGenerator
from generative_poetry.long_poem_generator2 import LongPoemGeneratorCore2


def get_user_id(update: Update) -> str:
    user_id = str(update.message.from_user.id)
    return user_id


def render_poem_html(poem_txt):
    # 03.02.2022 для отрисовки текста сгенерированного стиха в HTML формате в телеграмме
    s = '<pre>' + poem_txt + '</pre>'
    return s


def render_error_html(error_text):
    s = '<pre>🥵\n' + error_text + '</pre>'
    return s


temperature = 0.90
top_p = 1.00
top_k = 0
typical_p = 0.6


LIKE = 'Нравится!'
DISLIKE = 'Плохо :('
NEW = 'Новая тема'
MORE = 'Еще...'

last_user_poems = dict()
last_user_poem = dict()
user_format = dict()


FORMAT__COMMON = 'Лирика'
FORMAT__1LINER = 'Однострочники'
FORMAT__2LINER = 'Двухстрочники'
FORMAT__POROSHKI = 'Пирожки и порошки'
FORMAT__RUBAI = 'Рубаи'
FORMAT__KID = "Для детей"
FORMAT__PHIL = "Философские"
FORMAT__HUM = "Юмор и сатира"
#FORMAT__MIST = "Мистика"   23.10.2022 мистика объединилась с лирикой в новой модели генерации длинных стихов
FORMAT__FOLK = "Частушки"


def on_start(update, context) -> None:
    user_id = get_user_id(update)
    logging.debug('Entering START callback with user_id=%s', user_id)

    # 08.05.2022 сбросим историю использованных затравок
    seed_generator.restart_user_session(user_id)

    intro_text = "Привет, {}!\n\n".format(update.message.from_user.full_name) + \
    "Я - бот для генерации стихов (версия от 05.03.2023).\n" + \
    "Теперь вводите тему - какое-нибудь существительное или сочетание прилагательного и существительного, например <i>задорная улыбка</i>, " + \
    "и я сочиню стишок с этими словами.\n\n" + \
    "Можете также задавать полную первую строку, например <i>У бурных чувств неистовый конец</i>, я попробую продолжить от нее.\n\n" + \
    "Либо выберите готовую тему - см. кнопки внизу.\n" + \
    "Кнопка [<b>Ещё</b>] выведет новый вариант стиха на заданную тему. Кнопка [<b>Новая тема</b>] выведет новые затравки."

    # Получаем порцию саджестов (обычно 3 штуки) под выбранный жанр, чтобы пользователю не пришлось напрягаться
    # с придумыванием затравок.
    seeds = seed_generator.generate_seeds(user_id, domain=user_format.get(user_id))
    keyboard = [seeds]
    reply_markup = ReplyKeyboardMarkup(keyboard,
                                       one_time_keyboard=True,
                                       resize_keyboard=True,
                                       per_user=True)

    context.bot.send_message(chat_id=update.message.chat_id, text=intro_text, reply_markup=reply_markup, parse_mode='HTML')
    logging.debug('Leaving START callback with user_id=%s', user_id)


date2visitors = collections.defaultdict(list)
date2likes = collections.Counter()
date2dislikes = collections.Counter()
date2errors = collections.Counter()
date2prompts = collections.Counter()


def register_visitor(user_id):
    date2visitors[datetime.date.today()].append(user_id)


def on_stat(update, context) -> None:
    logging.info('Entering on_stat')
    # Вывод статистики по боту
    stat_lines = []

    stat_lines.append(f'Время аптайма, дней: {len(date2visitors)}')

    stat_lines.append(f'Всего затравок с момента старта: {sum(date2prompts.values())}')

    uniq_users = set(itertools.chain(*date2visitors.values()))
    stat_lines.append(f'Всего уникальных посетителей с момента старта: {len(uniq_users)}')

    stat_lines.append(f'Кол-во затравок сегодня: {date2prompts[datetime.date.today()]}')

    uniq_today = len(set(date2visitors[datetime.date.today()]))
    stat_lines.append(f'Кол-во уникальных посетителей сегодня: {uniq_today}')

    total_likes = sum(date2likes.values())
    stat_lines.append(f'Всего лайков с момента старта: {total_likes}')

    total_dislikes = sum(date2dislikes.values())
    stat_lines.append(f'Всего дизлайков с момента старта: {total_dislikes}')

    stat_lines.append(f'Кол-во лайков сегодня: {date2likes[datetime.date.today()]}')
    stat_lines.append(f'Кол-во дизлайков сегодня: {date2dislikes[datetime.date.today()]}')

    stat_lines.append(f'Кол-во ошибок сегодня: {date2errors[datetime.date.today()]}')

    context.bot.send_message(chat_id=update.message.chat_id, text='\n\n'.join(stat_lines))


def dbg_actions(update, context):
    pass # TODO ...
    return


def echo_on_error(context, update, user_id):
    date2errors[datetime.date.today()] += 1
    keyboard = [seed_generator.generate_seeds(user_id, domain='лирика')]
    reply_markup = ReplyKeyboardMarkup(keyboard,
                                       one_time_keyboard=True,
                                       resize_keyboard=True,
                                       per_user=True)

    context.bot.send_message(chat_id=update.message.chat_id,
                             text=render_error_html('К сожалению, произошла внутренняя ошибка на сервере, поэтому выполнить операцию не получилось.\nВыберите тему для новой генерации из предложенных или введите свою.'),
                             reply_markup=reply_markup, parse_mode='HTML')
    return


def echo(update, context):
    try:
        user_id = get_user_id(update)
        register_visitor(user_id)

        format = 'лирика'

        if update.message.text == NEW:
            # Пользователь хочет, чтобы ему предложили новые саджесты для генерации.
            last_user_poem[user_id] = None
            last_user_poems[user_id] = []

            keyboard = [seed_generator.generate_seeds(user_id, domain=format)]
            reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True, per_user=True)
            context.bot.send_message(chat_id=update.message.chat_id,
                                     text="Выберите тему из предложенных или введите свою",
                                     reply_markup=reply_markup)
            return

        if update.message.text == LIKE:
            if user_id not in last_user_poem:
                echo_on_error(context, update, user_id)
                return

            date2likes[datetime.date.today()] += 1

            # Какой текст полайкали:
            poem = last_user_poem[user_id].replace('\n', ' | ')
            logging.info('LIKE: poem="%s" user="%s"', poem, user_id)

            if len(last_user_poems[user_id]):
                keyboard = [[NEW, MORE]]
            else:
                keyboard = [[NEW]]

            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id, text="Спасибо :)", reply_markup=reply_markup)
            return

        if update.message.text == DISLIKE:
            if user_id not in last_user_poem:
                echo_on_error(context, update, user_id)
                return

            date2dislikes[datetime.date.today()] += 1

            # Какой текст не понравился:
            poem = last_user_poem[user_id].replace('\n', ' | ')
            logging.info('DISLIKE: poem="%s" user="%s"', poem, user_id)

            if len(last_user_poems[user_id]):
                keyboard = [[NEW, MORE]]
            else:
                keyboard = [[NEW]]

            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id, text="Понятно. Жаль :(", reply_markup=reply_markup)
            return

        if update.message.text == MORE:
            # Выведем следующее из уже сгенерированных

            if user_id not in last_user_poem or len(last_user_poems[user_id]) < 1:
                echo_on_error(context, update, user_id)
                return

            poem = last_user_poems[user_id][-1]

            last_user_poem[user_id] = poem
            last_user_poems[user_id] = last_user_poems[user_id][:-1]

            if len(last_user_poems[user_id]):
                keyboard = [[LIKE, DISLIKE, MORE, NEW]]
            else:
                keyboard = [[LIKE, DISLIKE], seed_generator.generate_seeds(user_id, domain=format)]

            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id,
                                     text=render_poem_html(last_user_poem[user_id]),
                                     reply_markup=reply_markup, parse_mode='HTML')

            return

        seed = update.message.text
        logging.info('Start generating the poems using seed="%s" for user="%s" id=%s in chat=%s', seed, update.message.from_user.name, user_id, str(update.message.chat_id))
        date2prompts[datetime.date.today()] += 1

        # 22.10.2022 Если генерация ничего не дала (например, все сгенерированные варианты не прошли фильтры),
        # то увеличиваем температуру и повторяем.
        temperature = 0.9
        max_temperature = 1.6

        while temperature <= max_temperature:
            ranked_poems = long_poetry_generator.generate_poems(topic=seed, temperature=temperature, top_p=top_p,
                                                          top_k=top_k, typical_p=typical_p, num_return_sequences=10)

            ranked_poems = sorted(ranked_poems, key=lambda z: -z[1])[:10]

            poems2 = [('\n'.join(lines), score) for lines, score in ranked_poems]

            if len(poems2) > 0:
                break

            temperature *= 1.1
            logging.info('Rising temperature to %f and trying again with seed="%s" for user="%s" id=%s in chat=%s', temperature, seed, update.message.from_user.name, user_id, str(update.message.chat_id))

        if len(poems2) == 0:
            logging.info('Could not generate a poem for seed="%s" for user="%s" id=%s in chat=%s', seed, update.message.from_user.name, user_id, str(update.message.chat_id))

        last_user_poems[user_id] = []
        last_user_poem[user_id] = None

        for ipoem, (poem, score) in enumerate(poems2, start=1):
            if ipoem == 1:
                last_user_poem[user_id] = poem
            else:
                last_user_poems[user_id].append(poem)

        if last_user_poem[user_id]:
            if len(last_user_poems[user_id]):
                keyboard = [[LIKE, DISLIKE, MORE, NEW]]
            else:
                keyboard = [[LIKE, DISLIKE], seed_generator.generate_seeds(user_id, domain=format)]

            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id,
                                     text=render_poem_html(last_user_poem[user_id]),
                                     reply_markup=reply_markup, parse_mode='HTML')
        else:
            keyboard = [seed_generator.generate_seeds(user_id, domain=format)]
            reply_markup = ReplyKeyboardMarkup(keyboard,
                                               one_time_keyboard=True,
                                               resize_keyboard=True,
                                               per_user=True)

            context.bot.send_message(chat_id=update.message.chat_id,
                                     text='Что-то не получается сочинить 😞\nЗадайте другую тему, пожалуйста',
                                     reply_markup=reply_markup)

    except Exception as ex:
        logging.error('Error in "echo"')
        logging.error(ex)
        logging.error(traceback.format_exc())
        echo_on_error(context, update, user_id)



def ngrams(s, n):
    return set(''.join(z) for z in zip(*[s[i:] for i in range(n)]))


def jaccard(s1, s2, shingle_len):
    shingles1 = ngrams(s1.lower(), shingle_len)
    shingles2 = ngrams(s2.lower(), shingle_len)
    return float(len(shingles1 & shingles2))/float(len(shingles1 | shingles2) + 1e-6)


def tokenize(s):
    return re.split(r'[.,!?\- ;:]', s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Verslibre & haiku generator v.18')
    parser.add_argument('--token', type=str, default='', help='Telegram token')
    parser.add_argument('--mode', type=str, default='console', choices='console telegram web'.split(), help='Frontend selector')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host for web frontend')
    parser.add_argument('--port', type=int, default=7860, help='Port for web frontend')
    parser.add_argument('--poetry_model', type=str, default='../../tmp/verses_generator_medium.chitalnya', help='Poetry generative model name path')
    parser.add_argument('--tmp_dir', default='../../tmp', type=str)
    parser.add_argument('--data_dir', default='../../data', type=str)
    parser.add_argument('--models_dir', default='../../models', type=str)
    parser.add_argument('--log', type=str, default='../../tmp/stressed_gpt_poetry_generation.{HOSTNAME}.{DATETIME}.log')

    args = parser.parse_args()
    mode = args.mode
    tmp_dir = os.path.expanduser(args.tmp_dir)
    models_dir = os.path.expanduser(args.models_dir)
    data_dir = os.path.expanduser(args.data_dir)

    init_logging(args.log, True)

    # 19-03-2022 запрещаем тензорфлоу резервировать всю память в гпу по дефолту, так как
    # это мешает потом нормально работать моделям на торче.
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    # Генератор саджестов
    seed_generator = SeedGenerator(models_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info('device=%s', str(device))

    # Транслятор затравок
    #prompt_builder = PromptBuilder(device)
    #prompt_builder.load_model('inkoziev/sbert_synonymy')
    #prompt_builder.load_prompts(os.path.join(data_dir, 'poetry', 'prompts', 'new_prompts.txt'))

    # Генератор рифмованных стихов
    long_poetry_generator = LongPoemGeneratorCore2(device)
    long_poetry_generator.load(gpt_model_path=args.poetry_model, models_dir=models_dir, data_dir=data_dir, tmp_dir=tmp_dir)

    if args.mode == 'telegram':
        # Телеграм-версия генератора
        telegram_token = args.token
        if len(telegram_token) == 0:
            telegram_token = getpass.getpass('Enter Telegram token:> ').strip()

        logging.info('Starting telegram bot')
        tg_bot = telegram.Bot(token=telegram_token).getMe()
        bot_id = tg_bot.name
        logging.info('Telegram bot "%s" id=%s', tg_bot.name, tg_bot.id)

        updater = Updater(token=telegram_token)
        dispatcher = updater.dispatcher

        start_handler = CommandHandler('start', on_start)
        dispatcher.add_handler(start_handler)

        stat_handler = CommandHandler('stat', on_stat)
        dispatcher.add_handler(stat_handler)

        echo_handler = MessageHandler(Filters.text & ~Filters.command, echo)
        dispatcher.add_handler(echo_handler)

        command_handler = MessageHandler(Filters.all, dbg_actions)
        dispatcher.add_handler(command_handler)
        #dispatcher.add_handler(CallbackQueryHandler(dbg_actions))

        #updater.dispatcher.add_handler(CallbackQueryHandler(format_menu, pattern='format=(.*)'))

        logging.getLogger('telegram.bot').setLevel(logging.INFO)
        logging.getLogger('telegram.vendor.ptb_urllib3.urllib3.connectionpool').setLevel(logging.INFO)

        logging.info('Start polling messages for bot %s', tg_bot.name)
        updater.start_polling()
        updater.idle()
    elif args.mode == 'web':
        import gradio as gr

        logging.info('Starting web UI on %s:%d', args.host, args.port)

        web_sessions = collections.defaultdict(dict)

        def web_generate(seed, session_id):
            user_id = str(session_id or 'web_guest')
            register_visitor(user_id)
            date2prompts[datetime.date.today()] += 1

            temperature_local = 0.9
            max_temperature = 1.6
            poems2 = []

            while temperature_local <= max_temperature:
                ranked_poems = long_poetry_generator.generate_poems(topic=seed, temperature=temperature_local, top_p=top_p,
                                                                    top_k=top_k, typical_p=typical_p, num_return_sequences=10)
                ranked_poems = sorted(ranked_poems, key=lambda z: -z[1])[:10]
                poems2 = [('\n'.join(lines), score) for lines, score in ranked_poems]
                if poems2:
                    break
                temperature_local *= 1.1

            if not poems2:
                date2errors[datetime.date.today()] += 1
                web_sessions[user_id]['current'] = None
                web_sessions[user_id]['rest'] = []
                return 'Что-то не получается сочинить 😞\nЗадайте другую тему, пожалуйста'

            web_sessions[user_id]['current'] = poems2[0][0]
            web_sessions[user_id]['rest'] = [p for p, _ in poems2[1:]]
            return web_sessions[user_id]['current']

        def web_more(session_id):
            user_id = str(session_id or 'web_guest')
            rest = web_sessions[user_id].get('rest', [])
            if not rest:
                return 'Больше готовых вариантов пока нет. Сгенерируйте стих по новой теме.'
            poem = rest.pop(0)
            web_sessions[user_id]['current'] = poem
            web_sessions[user_id]['rest'] = rest
            return poem

        def web_like(session_id):
            user_id = str(session_id or 'web_guest')
            poem = web_sessions[user_id].get('current')
            if poem:
                date2likes[datetime.date.today()] += 1
                return 'Спасибо :)'
            return 'Сначала сгенерируйте стих.'

        def web_dislike(session_id):
            user_id = str(session_id or 'web_guest')
            poem = web_sessions[user_id].get('current')
            if poem:
                date2dislikes[datetime.date.today()] += 1
                return 'Понятно. Жаль :('
            return 'Сначала сгенерируйте стих.'

        with gr.Blocks(title='VersLibre Poetry Generator') as demo:
            gr.Markdown('## VersLibre: генератор стихов\nВведите тему и получите стихотворение.')
            session_id = gr.Textbox(label='Session id (опционально)', value='web_guest')
            seed = gr.Textbox(label='Тема')
            output = gr.Textbox(label='Стих', lines=12)
            status = gr.Textbox(label='Статус')

            with gr.Row():
                generate_btn = gr.Button('Сгенерировать')
                more_btn = gr.Button('Ещё')
                like_btn = gr.Button('Нравится!')
                dislike_btn = gr.Button('Плохо :(')

            generate_btn.click(web_generate, inputs=[seed, session_id], outputs=[output])
            more_btn.click(web_more, inputs=[session_id], outputs=[output])
            like_btn.click(web_like, inputs=[session_id], outputs=[status])
            dislike_btn.click(web_dislike, inputs=[session_id], outputs=[status])

        demo.launch(server_name=args.host, server_port=args.port)
    else:
        print('Вводите затравку для генерации\n')

        while True:
            suggest = input(':> ').strip()

            logging.info('Generation for suggest "%s"...', suggest)
            ranked_poems = long_poetry_generator.generate_poems(topic=suggest, temperature=temperature, top_p=top_p,
                                                      top_k=top_k, typical_p=typical_p, num_return_sequences=10)

            ranked_poems = sorted(ranked_poems, key=lambda z: -z[1])

            # TODO - сделать ранжировку

            for poem, score in ranked_poems[:10]:
                print('\nscore={}'.format(score))
                print('-'*80)
                for line in poem:
                    print(line)
                print()
            print('='*50)
