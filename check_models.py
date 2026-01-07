import google.generativeai as genai

# Вставь свой ключ сюда для проверки
API_KEY = "AIzaSyCpbsu-H0ufb7jiFkkO_3C6QSwRuSTJOyE"

genai.configure(api_key=API_KEY)

print("Список доступных моделей:")
for m in genai.list_models():
    # Нам нужны только те модели, которые умеют генерировать текст (generateContent)
    if 'generateContent' in m.supported_generation_methods:
        print(f"- {m.name}")