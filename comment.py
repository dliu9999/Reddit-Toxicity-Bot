import praw
#import config
import os
from RNN_Model import predict_sentiment as ps

def bot_login():
    print("Logging in...")
    try:
        # r = praw.Reddit(username = config.username,
        #             password = config.password,
        #             client_id = config.client_id,
        #             client_secret = config.client_secret,
        #             user_agent = "Aggression comment responder v0.1")
        user = os.environ.get("username")
        passw = os.environ.get("password")
        cID = os.environ.get("client_id")
        secret = os.envornment.get("client_secret")
        r = praw.Reddit(username = user,
                    password = passw,
                    client_id = cID,
                    client_secret = secret,
                    user_agent = "Aggression comment responder v0.1")
        print("Logged In!")
    except:
        print("Failed to Log In!")
    
    return r

def run_bot(r):
    for comment in r.subreddit("test_for_lign_proj").comments(limit = 25):
        if "/u/BeNicePlease_Bot" in comment.body and comment.id not in comments_replied_to:
            print("Found!")

            parent = comment.parent().body
            prob = round(ps(parent), 3)

            if prob > 0.50:
                message = "You gotta be nicer than that man :)"
            else:
                message = "Nothing to see here :)"

            resp = "Score: " + str(prob) + ". " + message
            comment.reply(resp)
            print("Replied!")

            comments_replied_to.append(comment.id)


comments_replied_to = []

r = bot_login()
run_bot(r)
print("Success!")
