import csv
import operator
import re
import spacy

from collections import Counter
from sklearn.model_selection import train_test_split
from time import time

class NaiveBayesClassifier():
    '''
    ML classifier using Naive Bayes Classifier algoritm.
    '''
    def __init__(self):
        spacy_nlp = spacy.load('en_core_web_sm')
        print(f'spaCy Version: {spacy.__version__}')

        self.stop_words = spacy.lang.en.stop_words.STOP_WORDS
        self.classes = ['positive', 'negative', 'neutral']

    def parse(self, f, delimiter=',', class_index=1, confidence_index=2, text_index=10, use_confidence=True, negate=False, sentiment140=False):
        '''
        Parses CSV file and returns a tuple (text, confidence) and target-class.\n
        Default
        delimiter = ',' (comma), class_index = 1, confidence_index = 2, text_index = 10.\n
        If use_confidence=False, confidence is set to 1 for all text.\n
        If negate=True, every word in the part of a sentence containing a negatory word is appended 'NOT_' to the beginning.\n
        If sentiment140=True, the program extends dataset with 1.6 million tweets from Sentiment140 program - this is a work in progress and does not improve accuracy at this point.
        '''
        self.tweet_list = {'data': [], 'target': []}
        self.sentiment_counter = {'positive': 0, 'negative': 0, 'neutral': 0}

        with open(f) as csv_content:
            content = list(csv.reader(csv_content, delimiter=delimiter))
            for tweet in content[1:]:
                if negate:
                    text = self.check_negation(tweet[text_index])
                else:
                    text = tweet[text_index]
                # Format = (text, confidence)
                confidence = tweet[confidence_index] if use_confidence else 1
                self.tweet_list['data'].append((text, confidence))
                
                sentiment = tweet[class_index]
                self.sentiment_counter[sentiment] += float(confidence)
                self.tweet_list['target'].append(sentiment)
            
        if sentiment140:
            #Data downloaded from: http://help.sentiment140.com/for-students/
            with open('./trainingandtestdata/training.1600000.processed.noemoticon.csv', encoding='latin-1') as stanford_data:
                content = list(csv.reader(stanford_data, delimiter=','))

                sentiments = {'0': 'negative', '2': 'neutral', '4': 'positive'}

                for tweet in content:
                    # Data
                    if negate:
                        text = self.check_negation(tweet[-1])
                    else:
                        text = tweet[-1]
                    # Sentiment140 does not use confidence, thus 1 used as confidence for all text.
                    self.tweet_list['data'].append((text, 1))
                    
                    # Sentiment
                    sentiment = sentiments[tweet[0]]
                    self.sentiment_counter[sentiment] += 1
                    self.tweet_list['target'].append(sentiment)

        return self.tweet_list['data'], self.tweet_list['target']

    def check_negation(self, text):
        '''
        Splits text on delimiters and adds 'NOT_' to words (that are not in negatives) in part of text containing negative word.
        '''        
        negatives = ["n't", 'no', 'not', 'neither', 'nor']
        delimiters = ',|.|...|;|:|!|?'
        # Split on delimiters
        split_text = text.split(delimiters)
        new_text = ''
        # If a negatory word in text
        if any(neg in split_text for neg in negatives):
            for partial in split_text:
                # If a negatory word in partial text
                if any(neg in partial for neg in negatives):
                    # Add NOT_ to words except words in negatives
                    new_text += ' '.join([word if word in negatives else
                                        word if word[-3:] == "n't" or word[-3:] == "nt" else 'NOT_' + word for word in partial.split()]) + '. '
                else:
                    new_text += partial + '. '
            return new_text.lower()
        return text.lower()


    def fit(self, features, target, stopwords=False):
        '''
        For each class compile a histogram of words.\n
        If stopwords=True, program removes words that are found in a predifined stopwords-list.
        '''
        self.word_count = {'positive': {}, 'negative': {}, 'neutral': {}}        
        for i, data in enumerate(features):
            text, confidence = data
            sentiment = target[i]
            for word in text.split():
                # Specified with stopwords argument, wether to include words found in self.stop_words
                if stopwords == True:
                    if word in self.stop_words:
                        continue
                # Remove unwanted symbols from text
                word = re.sub('[^A-Za-z0-9-_@]+', '', word.lower())
                # Increment counter for specified sentiment for word with amount corresonding to confidence
                self.word_count[sentiment].setdefault(word, 0)
                self.word_count[sentiment][word] += float(confidence)
        self.vocabulary_len = len(set(self.word_count['positive']).union(set(self.word_count['negative']), set(self.word_count['neutral'])))
        return self.word_count

    def calc_c_prob(self, c):
        '''Calculate probability of given class'''
        return self.sentiment_counter[c] / len(self.tweet_list)

    
    def calc_word_prob(self, word, c):
        '''Calculate probability of a word given class c'''
        return (self.word_count[c][word] + 1) / (len(self.word_count[c]) + self.vocabulary_len)

    def calculate(self, sentence):
        '''Calculates most likely class using Naive Bayes Classifier for a sentence, returns most likely class.'''
        sentence = sentence.lower()
        probs = {}
        #Total number of words for each class
        num_word_c = {'positive': sum(self.word_count['positive'].values()), 'negative': sum(self.word_count['negative'].values()), 'neutral': sum(self.word_count['neutral'].values())}
        
        # Calculate probability of a class given a tweet.
        for c in self.classes:
            sentence_prob = 1
            for word in sentence.split():
                if word in self.word_count[c]:
                    sentence_prob *= self.calc_word_prob(word, c)
            probs[c] = self.calc_c_prob(c) * sentence_prob / (num_word_c[c] + self.vocabulary_len) # **len(sentence.split()) # removing vastly improved accuracy, generalising?
        
        # Return most likely class
        return max(probs.items(), key=operator.itemgetter(1))[0]

    def test(self, features, target, stopwords=False):
        '''
        Fits model to test data, calculates most probable class and verifies wether it's correct or not. Prints number of correct, and accuracy denoted in percentage.\n
        If stopwords=True, program removes words that are found in a predifined stopwords-list.
        '''
        self.fit(features, target, stopwords=stopwords)
        correct = 0
        for i, data in enumerate(features):
            text, confidence = data
            prediction = self.calculate(text)
            if prediction == target[i]:
                correct += 1
        print(f'{correct} correct out of {len(target)}')
        print('Accuracy: ', (correct / len(target)) * 100)

def main():
    # Timer for program runtime
    start = time()
    
    file = './data.csv'
    # format: tweet_id,airline_sentiment,airline_sentiment_confidence,negativereason,negativereason_confidence,airline,airline_sentiment_gold,name,negativereason_gold,retweet_count,text,tweet_coord,tweet_created,tweet_location,user_timezone
    
    nbc = NaiveBayesClassifier()
    X, y = nbc.parse(file)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    nbc.test(X_test + X_train , y_test + y_train)

    # Timer for program runtime
    print(f'Time spent: {time() - start:.2f} sec')

if __name__ == "__main__":
    main()
