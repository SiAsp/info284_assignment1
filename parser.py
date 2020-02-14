import csv
import operator
import re


def parse(f):
    with open(f) as csv_content:
        content = list(csv.reader(csv_content, delimiter=','))
        # Format: ['airline_sentiment', 'text']
        tweet_list = []
        for tweet in content[1:]:
            current = []
            current.append(tweet[1])
            text = check_negation(tweet[10])
            current.append(text)
            tweet_list.append(current)
        return tweet_list


def check_negation(text):
    # Splits text on delimiters and adds 'NOT_' to words (that are not in negatives) in part of text containing negative word.
    negatives = ["n't", 'no', 'not', 'neither', 'nor']
    delimiters = ',|.|...|;|:|!|?'
    # If a negatory word in text
    if any(neg in text for neg in negatives):
        # Split on delimiters
        text = text.split(delimiters)
        new_text = ''
        for partial in text:
            # If a negatory word in partial text
            if any(neg in partial for neg in negatives):
                # Add NOT_ to words except words in negatives
                partial = ' '.join([word if word in negatives else
                                    word if word[-3:] == "n't" else 'NOT_' + word for word in partial.split()])
            new_text += partial + '. '
        return new_text
    return text


def compile_word_bunch(tweet_list):
    delimiters = ',|.|...|;|:|!|?|'
    word_count = {'positive': {}, 'negative': {}, 'neutral': {}}
    for sentiment, text in tweet_list:
        for word in text.split():
            word = re.sub('[\.|\?|!||(|)|~"]+', '', word.lower())
            #word = re.sub('[^A-Za-z0-9-_@]+', '', word)
            word_count[sentiment].setdefault(word, 0)
            word_count[sentiment][word] += 1
    return word_count


class NaiveBayesClassifier():
    def __init__(self):
        pass


def main():
    file = './data.csv'
    tweet_list = parse(file)
    bunch = compile_word_bunch(tweet_list)
    for sentiment, val in bunch.items():
        sorted_dct = sorted(val.items(), key=operator.itemgetter(1))
        print(sorted_dct)
# print(tweet_list)


if __name__ == "__main__":
    main()
