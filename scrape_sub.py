import praw
import argparse
import os
import pandas as pd

#TODO: filter based on date

def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_posts', help='number of posts to scrape comments from, for all choose -1', default=25)
    parser.add_argument('-s', '--subreddit', help='subreddit to scrape', default='conservative')
    parser.add_argument('-o', '--outfile', help='output file', default='output.csv')
    parser.add_argument('-t', '--threshold', help='upvote filter threshold', default=50)

    return vars(parser.parse_args())

args = handle_args()

if not os.path.isfile(args['outfile']):
    f = open(args['outfile'], 'w')
    f.close()

r = praw.Reddit(client_id='gyc27_K63myg4A',
                client_secret='9B6TDDbIP0TFg6UP_KV5L2CB7qY',
                username='lg6596',
                password='pekoe1234',
                user_agent='scraper by /u/lg6596')

sub = r.subreddit(args['subreddit'])
post_generator = sub.top(limit=int(args['num_posts']))

comments = []
for submission in post_generator:
    for top_level_comment in submission.comments:
        if not isinstance(top_level_comment, praw.models.Comment):
            continue
        elif top_level_comment.score <= int(args['threshold']):
            continue
        comments.append(top_level_comment.body)

columns = ['text', 'label']
df = pd.DataFrame(columns=columns)
df['text'] = comments
df['label'] = [args['subreddit']] * len(df['text'])
df.to_csv(args['outfile'], index=False)