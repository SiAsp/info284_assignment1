import csv
import operator
import re


class NaiveBayesClassifier():
    def __init__(self):
        self.classes = ['positive', 'negative', 'neutral']

    def parse(self, f):
        with open(f) as csv_content:
            content = list(csv.reader(csv_content, delimiter=','))
            self.sentiment_counter = {'positive': 0, 'negative': 0, 'neutral': 0}
            # Format: ['airline_sentiment', 'text']
            self.tweet_list = []
            for tweet in content[1:]:
                current = []
                sentiment = tweet[1]
                self.sentiment_counter[sentiment] += 1
                current.append(sentiment)
                text = self.check_negation(tweet[10])
                current.append(text)
                self.tweet_list.append(current)
            return self.tweet_list

    def check_negation(self, text):
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

    def compile_word_bunch(self, tweet_list):
        delimiters = ',|.|...|;|:|!|?|'
        self.word_count = {'positive': {},
                           'negative': {}, 'neutral': {}}

        for sentiment, text in tweet_list:
            for word in text.split():
                word = re.sub('[\.|\?|!||(|)|~"]+', '', word.lower())
                # word = re.sub('[^A-Za-z0-9-_@]+', '', word)
                self.word_count[sentiment].setdefault(word, 0)
                self.word_count[sentiment][word] += 1
        self.vocabulary_len = len(set(self.word_count['positive']).union(set(self.word_count['negative']), set(self.word_count['neutral'])))

        return self.word_count

    def calc_c_prob(self, c):
        return self.sentiment_counter[c] / len(self.tweet_list)

    
    def calc_word_prob(self, word):
        '''        self.count_word_pos = self.word_count['positive'][word]
        self.count_word_neg = self.word_count['negative'][word]
        self.count_word_neu = self.word_count['neutral'][word]
        '''
        word_probs = {}

        for clss in self.classes:
            word_probs[clss] = (self.word_count[clss][word] + 1) / (len(self.word_count[clss]) + self.vocabulary_len)

        return word_probs

    def calclulate(self, sentence):
        sentence = sentence.lower()
        probs = {}
        num_word_c = {'positive': sum(self.word_count['positive'].values()), 'negative': sum(self.word_count['negative'].values()), 'neutral': sum(self.word_count['neutral'].values())}
        for c in self.classes:
            sentence_prob = 1
            for word in sentence.split():
                if word in self.word_count[c].keys():
                    sentence_prob *= self.calc_word_prob(word)[c]
                    print(sentence_prob)
            probs[c] = self.calc_c_prob(c) * sentence_prob / (num_word_c[c] + self.vocabulary_len)**len(sentence.split())
            print(probs[c])
        return probs

def main():
    file = './data.csv'
    nbc = NaiveBayesClassifier()
    tweet_list = nbc.parse(file)
    bunch = nbc.compile_word_bunch(tweet_list)

    print(nbc.calclulate("I am flying your #fabulous #Seductive skies again!"))


if __name__ == "__main__":
    main()
