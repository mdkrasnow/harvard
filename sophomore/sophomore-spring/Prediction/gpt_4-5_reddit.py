#!/usr/bin/env python3
import praw
import re
import os
import datetime
from dotenv import load_dotenv


def main():
    load_dotenv()
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),         
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),   
        user_agent=os.getenv('REDDIT_USER_AGENT'),        
    )

    # Get today's date
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Define the search query for GPT models
    # We use a broader query to capture posts that might mention GPT-4.5
    query = '"GPT" OR "ChatGPT" OR "GPT-4"'

    # Search for posts in r/all from the past day
    posts = reddit.subreddit("all").search(query, sort='new', time_filter='day', limit=None)

    # Compile regex patterns for different GPT-4.5 variations (case-insensitive)
    patterns = {
        "GPT-4.5": re.compile(r'\bGPT-4\.5\b', flags=re.IGNORECASE),
        "GPT 4.5": re.compile(r'\bGPT\s+4\.5\b', flags=re.IGNORECASE),
        "GPT4.5": re.compile(r'\bGPT4\.5\b', flags=re.IGNORECASE),
        "GPT 4_5": re.compile(r'\bGPT\s+4_5\b', flags=re.IGNORECASE),
        "GPT4_5": re.compile(r'\bGPT4_5\b', flags=re.IGNORECASE),
        "GPT-4_5": re.compile(r'\bGPT-4_5\b', flags=re.IGNORECASE),
    }

    # Dictionary to track subreddit-specific mentions
    subreddit_mentions = {}
    
    # Initialize counts for each pattern
    counts = {model: 0 for model in patterns}
    total_mentions = 0
    
    # Track unique posts that mention GPT-4.5 (any variation)
    posts_with_mentions = set()

    # Iterate through each post returned by the search
    for post in posts:
        # Skip posts not from today
        post_date = datetime.datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d")
        if post_date != today:
            continue
            
        # Combine the title and selftext (if available) for analysis
        content = post.title
        if hasattr(post, "selftext") and post.selftext:
            content += " " + post.selftext

        # Check for matches of each pattern
        found_match = False
        for model, pattern in patterns.items():
            matches = pattern.findall(content)
            if matches:
                counts[model] += len(matches)
                total_mentions += len(matches)
                found_match = True
                
                # Track subreddit mentions
                subreddit_name = post.subreddit.display_name
                if subreddit_name not in subreddit_mentions:
                    subreddit_mentions[subreddit_name] = 0
                subreddit_mentions[subreddit_name] += len(matches)
        
        # If any GPT-4.5 variant was found, add post to our set
        if found_match:
            posts_with_mentions.add(post.id)

    # Print the results
    print(f"GPT-4.5 Mention Analysis for {today}:")
    print("=" * 50)
    print("\nVariation counts:")
    for model, count in counts.items():
        print(f"{model}: {count}")
    
    print("\nTotal mentions across all variations:", total_mentions)
    print("Number of unique posts with mentions:", len(posts_with_mentions))
    
    # Print top subreddits by mention count (top 10)
    print("\nTop subreddits by GPT-4.5 mentions:")
    top_subreddits = sorted(subreddit_mentions.items(), key=lambda x: x[1], reverse=True)[:10]
    for subreddit, count in top_subreddits:
        print(f"r/{subreddit}: {count}")


if __name__ == '__main__':
    main()