from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

#Loads emails about Baseball and Hockey from the Email News Group
emails = fetch_20newsgroups(categories = ["rec.sport.baseball", "rec.sport.hockey"])

#Creates a Training dataset from the Emails selecting the Train Sub-set and Shuffling it, and using a Random_State = 108 to make it repeatible
train_emails = fetch_20newsgroups(subset = "train", shuffle = True, random_state = 108)

#Creates a Test label from the Email selecting the Test Sub-set and Shuffling it, and using a Random_State = 108 to make it repeatible
test_emails = fetch_20newsgroups(subset = "test", shuffle = True, random_state = 108)

#Creates a Counter to count words in both dataset (Train Mails and Test Mails)
counter = CountVectorizer()
counter.fit(test_emails.data + train_emails.data)

#Counts and Transform the words in th Train dataset
train_counts = counter.transform(train_emails.data)

#Counts and Transform the words in th Test dataset
test_counts = counter.transform(test_emails.data)

#Creates a Multi-Nomial Naive Bayes Classifier
classifier = MultinomialNB()

#Trains the created model with train data and it's target
classifier.fit(train_counts, train_emails.target)

#Prints the Model Score using Test Data and Target
print(classifier.score(test_counts, test_emails.target))

#Prints the Model Predictions using Test Data
print(classifier.predict(test_counts))