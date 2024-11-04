# Fall Detection Service
The fall detection webpage stream ip rtsp camera to recognize human fall and display video stream.

## Prepare weight
Download weight [`yolov7-w6-pose.pt`](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt). 
And put into directory `internal/yolov7_pose/weights`

## Prepare telegram bot
Create a new telegram bot and add this bot to a group for alarm. 

Change `bot_token` and group `chat_id` by your bot token and group id in file `internal/notification/telegram_sender.py`

## Run python
python version >= 3.10

Install pytorch
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```
Install requirement
```bash
pip install -r requirements.txt
```
Run
```bash
python main.py
```