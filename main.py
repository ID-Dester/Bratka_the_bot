# made by ID_Dester with love
import discord
import discord.ext
from discord.ext import commands, tasks
import json
import os
from art import *
import asyncio
import random
from groq import Groq

# переменная для хранения данных в памяти
users_cache = {}

intents = discord.Intents.all()
client = discord.Client(intents=intents)
tree = discord.app_commands.CommandTree(client)

@client.event
async def on_ready():
    global users_cache
    await tree.sync()
    print(text2art("ready"))
    # запуск задачи автосохранения
    if not save_data_task.is_running():
        save_data_task.start()

  # поиск конфиг файла
if os.path.exists('config.json'):
    with open('config.json', 'r') as f:
        config = json.load(f)
    TOKEN = config.get('token')
    GROQ_KEY = config.get('groq_key')
else:
    print("Ошибка: Файл config.json не найден!")
    exit()

# проверка ключа нейронки
groq_client = None
if GROQ_KEY:
    groq_client = Groq(api_key=GROQ_KEY)
else:
    print("Внимание: Groq Key не найден!")

# сохранение данных в json раз в 5 минут
@tasks.loop(minutes=5)
async def save_data_task():
    with open('users.json', 'w', encoding='utf-8') as f:
        json.dump(users_cache, f, ensure_ascii=False, indent=2)
    print(text2art("User data has been saved (autosave)."))

# загрузка данных с json
if os.path.exists('users.json'):
    with open('users.json', 'r', encoding='utf-8') as f:
        try:
            users_cache = json.load(f)
        except json.JSONDecodeError:
            users_cache = {} # если файл пустой

# тест команда
@tree.command(name="ping", description="pong")
async def slash_command(interaction: discord.Interaction):
  await interaction.response.send_message("pong")

# вывод рейтинга по количеству сообщений
@tree.command(name="top", description="Responds with top 10 users")
async def top_command(interaction: discord.Interaction):
    await interaction.response.defer()
    guild_id = str(interaction.guild_id)
    if guild_id not in users_cache:
        await interaction.followup.send("На этом сервере еще нет статистики.")
        return
    # Сортировка
    current_guild_stats = users_cache[guild_id]
    sorted_people = sorted(current_guild_stats.items(), 
                           key=lambda item: item[1], 
                           reverse=True)[:10] 
    lines = []
    for user_id, count in sorted_people:
        user = client.get_user(int(user_id))
        if not user:
             try:
                user = await client.fetch_user(int(user_id))
             except:
                user = "Unknown User"
        
        lines.append(f"{user}: {count} сообщений")
    
    await interaction.followup.send('\n'.join(lines))

# помощь с выбором
@tree.command(name="difficult_choice", description="Selects a random word (enter separated by a comma)")
async def choice_command(interaction: discord.Interaction, choices: str):
    choices_array = [x.strip() for x in choices.split(',') if x.strip()]    
    if not choices_array:
        await interaction.response.send_message("Hey, you haven't entered a single word!", ephemeral=True)
        return
    winner = random.choice(choices_array)
    await interaction.response.send_message(f"From the options: *{', '.join(choices_array)}*\nMy choice **{winner}**")

# вопрос нейросети
@tree.command(name="ask", description="ask groq a question")
async def ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    if not groq_client:
        await interaction.followup.send("Нейросеть не подключена (нет ключа).")
        return
    try:
        # Отправляем запрос
        chat_completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant", 
            messages=[
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0.7, # Креативность (0.5 - строго, 1.0 - фантазия)
            max_tokens=1024, # Лимит длины ответа
        )

        answer = chat_completion.choices[0].message.content

        if len(answer) > 1900:
            answer = answer[:1900] + "... (ответ обрезан)"

        await interaction.followup.send(f"**Вопрос:** {question}\n**Llama:** {answer}")

    except Exception as e:
        print(f"Ошибка Groq: {e}")
        # Если ошибка 429 вылезет и тут, значит сервер перегружен, но это бывает редко
        await interaction.followup.send("Нейросеть сейчас недоступна, попробуй позже.")

# Счет сообщений на сервере
@client.event
async def on_message(message):
    if message.author.bot:
        return
    guild_id = str(message.guild.id)
    user_id = str(message.author.id)
    if guild_id not in users_cache:
        users_cache[guild_id] = {} 
    if user_id in users_cache[guild_id]:
        users_cache[guild_id][user_id] += 1
    else:
        users_cache[guild_id][user_id] = 1

if TOKEN:
    client.run(TOKEN)
else:
    print("Ошибка: Токен не найден в config.json")
