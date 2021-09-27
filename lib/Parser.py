FIELDS = ['id', 'lang', 'text', 'freqs']


class Parser:
    def __init__(self, filename):
        self.parse(self.read_file(filename))

    def parse(self, readfile):
        self.dataset = []

        for data in readfile:
            # Remove unnecessary characters and make a tuple
            parsed = tuple(map(str.strip, data.split('\t')))

            # Complete the data with unigram frequencies
            text = parsed[FIELDS.index('text')]
            entry = (*parsed, self.ngram_freq(text, 1))

            self.dataset.append(entry)

    def get(self, field):
        return [entry[FIELDS.index(field)] for entry in self.dataset]

    @staticmethod
    def ngram_freq(string, n, lc=True):
        COUNT = len(string) - n + 1

        if lc:
            string = string.lower()
            
        # Find the absolute frequency of each ngram
        ngrams = {}
        
        for i in range(COUNT):
            ngram = string[i:i+n]

            try:
                ngrams[ngram] += 1
            except KeyError:
                ngrams[ngram] = 1
                
        # Normalize absolute frequencies
        for ngram in ngrams:
            ngrams[ngram] /= COUNT
            
        return ngrams

    @staticmethod
    def read_file(filename):
        with open(filename, encoding='utf8') as file:
            return file.read().strip().splitlines()