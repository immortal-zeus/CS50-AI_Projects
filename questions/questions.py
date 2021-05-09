import nltk
import sys , os

FILE_MATCHES = 1
SENTENCE_MATCHES = 1


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    Mapped_dict= dict()
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(directory, filename) ,  encoding='utf-8') as f:
            contents = f.read()
            Mapped_dict[filename]=contents
    return Mapped_dict

def tokenize(document):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    import string
    final_word=list()
    for word in nltk.word_tokenize(document.lower()):
        if word not in string.punctuation and word not in nltk.corpus.stopwords.words("english"):
            final_word.append(word)
    return final_word



def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    import math
    final_dic_doc=dict()
    all_words = set()
    for doc in documents:
        all_words.update(set(documents[doc]))

    for word in all_words:
        count =0
        for doc in documents:
            if word in documents[doc]:
                count+=1
        final_dic_doc[word]=math.log(len(documents)/count)

    return final_dic_doc

def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    tf_idf=dict()
    for file in files:
        tf_idf_values =0
        for word in query:
            tf = files[file].count(word)
            if word in idfs:
                tf_idf_values += (tf * idfs[word])
            else:
                print("##Please Note: Some words in the query were not present in the idfs.")
        tf_idf[file]=tf_idf_values

    return sorted(tf_idf, key=lambda x: tf_idf[x], reverse=True)[:n]

def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_idf=dict()
    for word in query:
        for sentence in sentences:
            if word in sentences[sentence]:
                if sentence in sentence_idf:
                    if word in idfs:
                        sentence_idf[sentence]+=idfs[word]
                    else:
                        print("##Please Note: Some words in the query were not present in the idfs.")
                else:
                    if word in idfs:
                        sentence_idf[sentence]=idfs[word]
                    else:
                        print("##Please Note: Some words in the query were not present in the idfs.")



    return sorted(sentence_idf,key=lambda x:(sentence_idf[x],term_density(x,query)),reverse=True)[:n]


def term_density(sentence , query):
    count=0
    for s_word  in sentence.split():
        if s_word in query:
            count+=1

    return count/len(sentence.split())


if __name__ == "__main__":
    main()
