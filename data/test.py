# Read file and find original tweet (non-RT)

with open("data/raw/output3.txt", "r", encoding="utf-8") as file:
    for line in file:
        # Split by tab (your data appears tab-separated)
        columns = line.strip().split("\t")
        
        if len(columns) > 2:
            tweet_text = columns[2]
            
            # Check if it's NOT a retweet
            if not tweet_text.startswith("RT @"):
                print("Original Tweet Found:\n")
                print(line)