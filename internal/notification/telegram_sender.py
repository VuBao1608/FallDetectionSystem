from datetime import datetime

import requests
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import threading
from telegram import Bot

# bot_token = "7533380319:AAF1jUoiw-J8WocrAMgquJlAvz6C-CrkpJw"
bot_token = "7200455173:AAEvXDj5Z7rHvvRYqJKxAqARIGsJnDE7FgA"
chat_id = "-4562755988"
# chat_id = "-4537263169"
bot = Bot(token=bot_token)


# Function to send image via Telegram using requests with a caption
def send_image_with_caption(image: np.ndarray, caption: str, scale=None):
    print('Sending image caption')
    # Convert NumPy array to Pillow Image
    h, w = image.shape[:2]

    if scale is not None:
        res_image = cv2.resize(image, (int(w * scale), int(h * scale)))
    else:
        res_image = image

    res_image = res_image[..., ::-1]
    pil_image = Image.fromarray(res_image.astype('uint8'))

    # Save image to a bytes buffer
    buffer = BytesIO()
    pil_image.save(buffer, format='JPEG')
    buffer.seek(0)

    # Send the image as a multipart/form-data request with caption
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto?caption={caption}"
    files = {'photo': buffer}
    data = {'chat_id': chat_id,
            'parse_mode': 'MarkdownV2'}

    response = requests.post(url, data=data, files=files, timeout=3)

    # Check if the request was successful
    if response.status_code == 200:
        print("Image sent successfully!")
    else:
        print(f"Failed to send image: {response.status_code}")
        print(response.text)

    # bot.send_photo(chat_id=chat_id, photo=buffer, caption=caption)
    print('Sent notification to telegram')


def send_alarm_telegram(image: np.ndarray):
    print('Send telegram message')
    now = datetime.now()
    str_now = now.strftime("%m/%d/%Y %H:%M:%S")
    message = "⚠️⚠️⚠️ *Warning\\!* There is a person falls at " + str_now

    t = threading.Thread(target=send_image_with_caption, args=(image, message, None))
    t.start()
