import csv
import re


def parse(f):
    with open(f) as csv_content:
        content = list(csv.reader(csv_content, delimiter=','))
        # Format: ['airline_sentiment', 'airline_sentiment_confidence', 'negativereason', 'negativereason_confidence', 'text']
        tweets = []
        for tweet in content[1:]:
            current = []
            [current.append(tweet[i]) for i in range(1, 5)]
            text = check_negation(tweet[10])
            current.append(text)
            tweets.append(current)

        return tweets,


def check_negation(text):
    # Splits text on delimiters and adds 'NOT_' to words (that are not in negatives) in part of text containing negative word.
    negatives = ["n't", 'not', 'never', 'neither', 'nor', 'nothing']
    delimiters = [',', '.', ';', ':', '!', '?']
    # If a negatory word in text
    if any(neg in text for neg in negatives):
        # Split on delimiters
        regexPattern = '|'.join(map(re.escape, delimiters))
        text = re.split(regexPattern, text)
        new_text = ''
        for partial in text:
            # If a negatory word in partial text
            if any(neg in partial for neg in negatives):
                # Add NOT_ to words except words in negatives
                partial = ' '.join(
                    ['NOT_' + word if word not in negatives else 'NOT_' + word if word[-3:] != "n't" else word for word in partial.split()])
            new_text += partial + '. '
        return new_text
    return text


class NaiveBayesClassifier():
    def __init__(self):
        pass


def main():
    tweet_list = parse(
        '/Users/asplem/Documents/Infovit/info284/oblig/oblig1/data.csv')
    print(tweet_list)


if __name__ == "__main__":
    main()

'''
P(word) * P(word | +) for all words in tweet /
= > P()
'''
