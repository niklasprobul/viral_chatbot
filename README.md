# Viral Mock Chatbot
Mock Telegram Chatbot for the Viral (2022) Call of Cthulhu Campaign

## Quick Start
1. Initialize a conda env with the provided `requirements.txt`.
2. Add the following keys to `data/secrets.json` (e.g. by modifying `data/template_secrets.json`).
- `token`: in Telegram Create a bot using BotFather and use the provided token (str)
- `admin_id` : Get you Telegram userid (e.g. by messaging @RawDataBot) (int)
- `chat_id` : Create a Telegram Supergroup (e.g. by enabling topics), and create the topics '`chat`', '`donations`', '`leaderboard`', and '`maya`'. Send a message in each chat. Right click -> copy message link to get the ids in the following format: `https://t.me/c/[chat_id]/[topic_id]/[message_id]]`. Use the `chat_id` here and put the `topic_ids` into the chat_bot.py. (int)

3. Run pyhton3 `./chat_bot.py`
4. In a Telegram 1on1 conversation with your bot (`@[the name you set in BotFather]`) run the `/start` command. 
5. Control the Bot via the interface or `/` commands (run `/help` for a list). 
6. Chats and Donation are simulated based on a mood and location and pulled from the respective files in data. Press `Resume` for the bot to start the chat simulation.

## Controls
The chatbot can be controled by the admint (set by admin_id) in a private chat. The control panel is generated based on the provided moods and locations. 
```
[ Mood Block ] // select current mood (1 required)
[ Location Block] // select current location (1 required)
[ Pause / Resume ]  // pause / resume the chatbot (starts in pause)
[ Send Maya ]    // type message that is sent in the maya channel
[ Send Special ]    // type message that is sent as the 'special user' in the chat
[ Send donation ]    // type name, then type amount sent as donation with random text
[ Burst Subs ]    // type number (can be negative) that is added to the subscriber count
[ Burst Viewers ]    // type number (can be negative) that is added to the viewer count
[ Spam ]    // type message that is spammed in varied forms in the chat for 20 ticks
[ Reload Config ] // Reloads configs
[ Init Leaderboard ] // Starts a new leaderboard in the respective channel
[ Refresh Board ] // Manually refresh leaderboard (automativally once per minute)
[ Hype - ] Slows all chats and donations by 20%
[ Hype + ] Speeds up all chats and donations by 20% (up to 500%)
```

## Config
### Moods
Every mood can be configured individually in `data/mood_config.json`. To add you custom mood, add it in the config and 
```jsonc
"good": { // name of the mood
  "mood_interval": 0.8, // interval multiplicator for sending chats / donations
  "donation_range": [5.0, 50.0], // range of donation amount. Weighted towards the lower value
  "audience_trend": 1 // trend if audience increases or decreased. Set to -1 if you want the mood to decrease audience
},
```

### Locations
Every location can be configured individually in `data/location_scripts.json`.
```jsonc
"stream_start": { // location name
    "audience_hooks": [ // guaranteed chats from random users in the chat channel
      "Lets gooooo!",
      "FIRST"
    ],
    "maya_lines": [ // guaranteed messages in the maya channel
      "We are live!",
      "Please double check all feeds",
      "And dont forget your role call, babes!"
    ],
    "donations": [ // guaranteed donations 
      {
        "donor": "ultimaunum",
        "amount": 10,
        "note": "welcome!"
      }
    ],
    "donation_level": 0.11, // trigger a donation every 1/donation_level ticks [0.0 to 1.0]
    "subs_per_minute": 30, // avg subscriber gain per minute (can be negative)
    "viewers_per_minute": 30 // avg viewer gain per minute (can be negative)
  },
```

### Audience 
The viewers and subscribers can be configured to have a start and limit (where it will +- hover) in `data/audience_config.json`.
```jsonc
{
  "subscribers_start": 950011,
  "subscribers_plateau": 998655,
  "viewers_start": 301,
  "viewers_plateau": 1000000
}
```

