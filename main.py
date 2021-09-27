from lib.Parser import Parser
from lib.Classifier import Classifier


parser = Parser("sentences.tsv")

classifier = Classifier(parser)
classifier.train()

print("The classifier can predict languages with an accuracy of %.4f" % classifier.test())

while (text := input('[ ]: ')) != 'exit':
    print(classifier.predict(classifier.prepare(text), readable=True)[0])
