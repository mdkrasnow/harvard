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

    # Fetch top posts in r/all from the past day.
    posts = reddit.subreddit("all").top(time_filter='day', limit=1000)

    # Compile regex patterns (case-insensitive) for each AI model.
    patterns = {
        "ChatGPT": re.compile(r'\bchatgpt\b', flags=re.IGNORECASE),
        "Gemini":  re.compile(r'\bgemini\b', flags=re.IGNORECASE),
        "Claude":  re.compile(r'\bclaude\b', flags=re.IGNORECASE),
        "DeepSeek": re.compile(r'\bdeepseek\b', flags=re.IGNORECASE)
    }

    # Initialize a dictionary to keep counts for each model.
    counts = {model: 0 for model in patterns}

    # Iterate through each post returned.
    for post in posts:
        # Use only the title (headline) for analysis.
        content = post.title

        # Count occurrences for each model in the current headline.
        for model, pattern in patterns.items():
            matches = pattern.findall(content)
            counts[model] += len(matches)

    # Determine which model has the maximum number of references.
    most_referenced = max(counts, key=counts.get)

    # Print out the counts and the most referenced AI model.
    print("AI model reference counts for today (headlines only):")
    for model, count in counts.items():
        print(f"{model}: {count}")
    print("\n====================================")
    print(f"The AI model referenced most in today's Reddit headlines is: {most_referenced} "
          f"(with {counts[most_referenced]} references)")

if __name__ == '__main__':
    main()
