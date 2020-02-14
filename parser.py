import csv
import operator
import re
from collections import Counter

from sklearn.model_selection import train_test_split

class NaiveBayesClassifier():
    def __init__(self):
        self.classes = ['positive', 'negative', 'neutral']

    def parse(self, f):
        with open(f) as csv_content:
            content = list(csv.reader(csv_content, delimiter=','))
            self.sentiment_counter = {'positive': 0, 'negative': 0, 'neutral': 0}
            # Format: ['airline_sentiment', 'text']
            self.tweet_list = {'data': [], 'target': []}
            for tweet in content[1:]:
                current = []
                text = tweet[10]
                #text = self.check_negation(tweet[10])
                self.tweet_list['data'].append(text)
                sentiment = tweet[1]
                self.sentiment_counter[sentiment] += 1
                self.tweet_list['target'].append(sentiment)
            return self.tweet_list['data'], self.tweet_list['target']

    def check_negation(self, text):
        # Splits text on delimiters and adds 'NOT_' to words (that are not in negatives) in part of text containing negative word.
        negatives = ["n't", 'no', 'not', 'neither', 'nor']
        delimiters = ',|.|...|;|:|!|?'
        # Split on delimiters
        text = text.split(delimiters)
        new_text = ''
        for partial in text:
            # If a negatory word in partial text
            if any(neg in partial for neg in negatives):
                # Add NOT_ to words except words in negatives
                new_text += ' '.join([word if word in negatives else
                                    word if word[-3:] == "n't" else 'NOT_' + word for word in partial.split()]) + '. '
            else:
                new_text += partial + '. '

        return new_text.lower()


    def fit(self, data, target):
        delimiters = ',|.|...|;|:|!|?|'
        self.word_count = {'positive': {},
                           'negative': {}, 'neutral': {}}
        for i, text in enumerate(data):
            sentiment = target[i]
            for word in text.split():
                word = re.sub('[^A-Za-z0-9-_@]+', '', word.lower())
                self.word_count[sentiment].setdefault(word, 0)
                self.word_count[sentiment][word] += 1
        self.vocabulary_len = len(set(self.word_count['positive']).union(set(self.word_count['negative']), set(self.word_count['neutral'])))
        return self.word_count

    def calc_c_prob(self, c):
        return self.sentiment_counter[c] / len(self.tweet_list)

    
    def calc_word_prob(self, word, c):
        return (self.word_count[c][word] + 1) / (len(self.word_count[c]) + self.vocabulary_len)

    def calclulate(self, sentence):
        sentence = sentence.lower()
        probs = {}
        num_word_c = {'positive': sum(self.word_count['positive'].values()), 'negative': sum(self.word_count['negative'].values()), 'neutral': sum(self.word_count['neutral'].values())}
        for c in self.classes:
            sentence_prob = 1
            for word in sentence.split():
                if word in self.word_count[c].keys():
                    sentence_prob *= self.calc_word_prob(word, c)
            probs[c] = self.calc_c_prob(c) * sentence_prob / (num_word_c[c] + self.vocabulary_len)**len(sentence.split())
        return max(probs.items(), key=operator.itemgetter(1))[0]

    def test(self, data, target):
        self.fit(data, target)
        correct = 0
        for i, text in enumerate(data):
            prediction = self.calclulate(text)
            if prediction == target[i]:
                correct += 1
        print(f'{correct} correct out of {len(target)}')
        print('Error-percentage: ', 1-(correct / len(target)) * 100)

def main():
    file = './data.csv'
    nbc = NaiveBayesClassifier()
    X, y = nbc.parse(file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=40)

    nbc.test(X_test, y_test)

if __name__ == "__main__":
    main()
