# Импорт необходимых модулей и классов
from gigachat.models import Chat, Messages, MessagesRole
from dotenv import load_dotenv
from gigachat import GigaChat
from gtts import gTTS
import threading
import tempfile
import pyaudio
import pygame
import vosk
import json
import time
import os

# Загрузка переменных окружения и получение учетных данных
load_dotenv()
credentials = os.getenv("CREDENTIALS")

# Устанавливаем путь для временного файла с распознанным текстом
output_file_path = os.path.join(tempfile.gettempdir(), "recognized_text.txt")

# Настройка Vosk для распознавания речи на русском языке
vosk.SetLogLevel(-1)
model = vosk.Model(lang="ru")
rec = vosk.KaldiRecognizer(model, 16000)

# Инициализация PyAudio для работы с аудио устройствами
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,
				channels=1,
				rate=16000,
				input=True,
				frames_per_buffer=8192)


# Функция для синтеза речи из текста
def msg(text):
	tts = gTTS(text=text, lang="ru", slow=False)
	rand = time.time()
	tts.save(os.path.join(tempfile.gettempdir(), f"{rand}_tts.mp3"))
	pygame.mixer.init()
	pygame.mixer.music.load(os.path.join(tempfile.gettempdir(), f"{rand}_tts.mp3"))
	pygame.mixer.music.play()


# Основная функция программы
def main():
	# Открываем файл для записи распознанных текстов
	with open(output_file_path, "a", encoding="utf-8") as output_file:
		print("Начало прослушивания.")

		# Запускаем поток для синтеза сообщения о готовности принимать ответы
		th = threading.Thread(target=msg, args=["готов принимать ответы"])
		th.start()
		th.join()

		# Основной цикл прослушивания
		while True:
			data = stream.read(4096)

			# Проверяем результат распознавания
			if rec.AcceptWaveform(data):
				result = json.loads(rec.Result())
				recognized_text = result['text']

				# Записываем распознанный текст в файл
				output_file.write(recognized_text + "\n")

				# Проверяем ключевые слова для взаимодействия с нейронным чатом
				if "нейро" in recognized_text.lower() or "вопрос" in recognized_text.lower() or "окей" in recognized_text.lower():
					# Создаем payload для запроса к нейронному чату
					payload = Chat(
						messages=[
							Messages(
								role=MessagesRole.SYSTEM,
								content=f"Ты ИИ бот для помощи пользователю. Ты работаешь в виде программы на ПК. "
										f"Текущая дата: {time.ctime(time.time())}"
										f"Пользователь ПК: {os.getlogin()}"
							),
							Messages(
								role=MessagesRole.USER,
								content=recognized_text
							)
						],
						temperature=0.5,
						max_tokens=500,
					)

					# Инициализируем GigaChat и отправляем запрос
					giga = GigaChat(credentials=credentials, verify_ssl_certs=False)
					r = giga.chat(payload)

					print(f"Нейро: {r.choices[0].message.content}")

					# Синтезируем и произносим ответ от нейронного чата
					threading.Thread(target=msg, args=[r.choices[0].message.content]).start()

				# Проверяем команду для остановки программы
				if "стоп" in recognized_text.lower() or "выход" in recognized_text.lower() or "выключить" in recognized_text.lower() or "остановка" in recognized_text.lower():
					print("Остановка...")
					break

		# Останавливаем и закрываем аудио поток
		stream.stop_stream()
		stream.close()

		# Завершаем работу PyAudio
		p.terminate()


if __name__ == '__main__':
	main()
