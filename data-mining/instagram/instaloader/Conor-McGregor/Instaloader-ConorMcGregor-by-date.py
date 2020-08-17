from datetime import datetime
from itertools import dropwhile, takewhile

import instaloader

L = instaloader.Instaloader()
PROFILE = 'mcgregormania'

posts = instaloader.Profile.from_username(L.context, PROFILE).get_posts()

SINCE = datetime(2019, 5, 1)
UNTIL = datetime(2019, 3, 1)

for post in takewhile(lambda p: p.date > UNTIL, dropwhile(lambda p: p.date > SINCE, posts)):
    print(post.date)
    L.download_post(post, PROFILE)