# Viral Mock Chatbot
Mock Telegram Chatbot for the Viral (2022) Call of Cthulhu Campaign

## Quick Start
1. Initialize a conda env with the provided requirements.txt
2. Add the following keys to data/secrets.json (e.g. by modifying data/template_secrets.json)
- `token`: in Telegram Create a bot using BotFather and use the provided token (str)
- `admin_id` : Get you Telegram userid (e.g. by messaging @RawDataBot) (int)
- `chat_id` : Create a Telegram Supergroup (e.g. by enabling topics), and create the topics '`chat`', '`donations`', '`leaderboard`', and '`maya`'. Send a message in each chat. Right click -> copy message link to get the ids in the following format: `https://t.me/c/[chat_id]/[topic_id]/[message_id]]`. Use the `chat_id` here and put the `topic_ids` into the chat_bot.py. (int)

3. Run pyhton3 ./chat_bot.py
4. In a Telegram 1on1 conversation with you bot (`@[the name you set in BotFather]`) run the `/start` command. 
5. Control the Bot via the interface or / commands (run `/help` for a list). 
6. Chats are simulated based on a mood and location and pulled from the respective files in data. Press `Resume` for the bot to start the chat simulation. 
