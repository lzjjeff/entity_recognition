import re
import nltk
from nltk.tag import StanfordNERTagger


train_data_path = './data/Genia4ERtask1.iob2'
test_data_path = './data/sampletest1.raw'


def read(path):
    fi = open(path, 'r')
    try:
        b = fi.read()
        # print(b)
    finally:
        fi.close()
    return b


'''
特征1：是否存在大写字母，是为1，否为0
'''
def add_feature1(text):
    newtext = ''
    text = text.split('\n')
    for line in text:
        if line != '':
            flag = 0
            word, ne = line.split('\t')
            # print(word, ne)
            if re.match('[A-Z]+', word):
                flag = 1
            newtext += ne + '\t' + word + '\t' + str(flag) + '\n'
    print(newtext)
    return newtext


def output_2_file(text, path):
    fo = open(path, 'w')
    try:
        fo.write(text)
    finally:
        fo.close()


def test_stanford_ner():
    testfile = open('./data/sampletest1.raw', 'r')
    text = testfile.read()
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    entities = nltk.chunk.ne_chunk(tagged, binary=False)
    print(entities)


if __name__ == '__main__':
    # text = read(train_data_path)
    # newtext = add_feature1(text)
    # output_2_file(newtext, 'feature1.txt')

    test_text = read(test_data_path)
    newtext = ''
    text = test_text.split('\n')
    for line in text:
        if line != '':
            newtext += 'O' + '\t' + line + '\n'
    output_2_file(newtext, 'test.txt')
