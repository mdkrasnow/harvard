#!/usr/bin/env python3
import praw
import re
import os
from dotenv import load_dotenv




def main():
    load_dotenv()
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_CLIENT_ID'),         
        client_secret=os.getenv('REDDIT_CLIENT_SECRET'),   
        user_agent=os.getenv('REDDIT_USER_AGENT'),        
    )

    # Define the search query.
    # The quotes force an exact phrase match. We combine the AI models with OR.
    query = '"ChatGPT" OR "Gemini" OR "Claude" OR "DeepSeek"'

    # Search for posts in r/all from the past day.
    posts = reddit.subreddit("all").search(query, sort='new', time_filter='day', limit=None)

    # Compile regex patterns (case-insensitive) for each AI model.
    patterns = {
        "ChatGPT": re.compile(r'\bchatgpt\b', flags=re.IGNORECASE),
        "Gemini":  re.compile(r'\bgemini\b', flags=re.IGNORECASE),
        "Claude":  re.compile(r'\bclaude\b', flags=re.IGNORECASE),
        "DeepSeek": re.compile(r'\bdeepseek\b', flags=re.IGNORECASE)
    }

    # Initialize a dictionary to keep counts for each model.
    counts = {model: 0 for model in patterns}

    # Iterate through each post returned by the search.
    for post in posts:
        # Combine the title and selftext (if available) for analysis.
        content = post.title
        if hasattr(post, "selftext") and post.selftext:
            content += " " + post.selftext

        # Count occurrences for each model in the current post.
        for model, pattern in patterns.items():
            matches = pattern.findall(content)
            counts[model] += len(matches)

    # Determine which model has the maximum number of references.
    most_referenced = max(counts, key=counts.get)

    # Print out the counts and the most referenced AI model.
    print("AI model reference counts for today:")
    for model, count in counts.items():
        print(f"{model}: {count}")
    print("\n====================================")
    print(f"The AI model referenced most on Reddit today is: {most_referenced} "
          f"(with {counts[most_referenced]} references)")

if __name__ == '__main__':
    main()
