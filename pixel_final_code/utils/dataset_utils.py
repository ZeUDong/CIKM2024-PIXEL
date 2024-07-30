
import re
from nltk.corpus import wordnet as wn
import random
import torch
from transformers.models.bert.tokenization_bert import BertTokenizer
import numpy as np

class dataset_text_processer:
    def __init__(self):   
        self.tokenizer = BertTokenizer.from_pretrained("../bert/")
        self.vocabs = list(self.tokenizer.vocab.keys())
        #self.max_words = 180#30 # test   cub
        #self.max_words = 180#30 # test   awa2
        self.max_words = 180#30 # test   sun


    def get_all_texts_labels(self, input_texts):
        input_ids_list, attention_mask_list, mask_labels_list, replace_labels_list = list(), list(), list(), list()
        origin_tokens_lengths = []
        for text in input_texts:
            input_ids, attention_mask, mask_labels, replace_labels,origin_tokens_length = self.get_text_train_labels(text)
            # print(text)
            # print(input_ids)
            # print(attention_mask)
            # print(mask_labels)
            # print(replace_labels)
            # bb
            origin_tokens_lengths.append(origin_tokens_length)
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            mask_labels_list.append(mask_labels)
            replace_labels_list.append(replace_labels)
        # print(np.min(origin_tokens_lengths),np.mean(origin_tokens_lengths),np.max(origin_tokens_lengths))
        # bb
        return input_ids_list, attention_mask_list, mask_labels_list, replace_labels_list


    def get_text_train_labels(self, input_text):
        description_tokens, des_mask_token, des_mask_sign, des_replace_token, des_replace_sign = \
            self.process_description(input_text)
        # print(description_tokens, des_mask_token, des_mask_sign, des_replace_token, des_replace_sign)

        all_tokens = description_tokens 
        mask_sign = des_mask_sign 
        # print(mask_sign)
        # bb
        replace_sign = des_replace_sign
        tokens_length = len(all_tokens)
        origin_tokens_length = len(all_tokens)
        #print(tokens_length)
        if tokens_length > self.max_words:
            all_tokens = all_tokens[:self.max_words]
            mask_sign  = mask_sign[:self.max_words]
            replace_sign = replace_sign[:self.max_words]

        pad_length = self.max_words - tokens_length
        tokens_length = len(all_tokens)
        pad_tokens = ['[PAD]'] * pad_length
        replace_sign = replace_sign + [0]*pad_length
        all_tokens = all_tokens + pad_tokens

        mask_token = des_mask_token 
        # print(des_mask_token)


        mask_token = mask_token[:sum(mask_sign)]
        # print(mask_token)
        # bbb
        input_ids = self.tokenizer.convert_tokens_to_ids(all_tokens)
        mask_token_ids = self.tokenizer.convert_tokens_to_ids(mask_token)

        mask_labels = [-100] * self.max_words
        # mask_labels = [0] * self.max_words
        mask_pos = [i for i,a in enumerate(mask_sign) if a ==1]

        assert len(mask_pos) == len(mask_token_ids)

        for p,l in zip(mask_pos, mask_token_ids):
            mask_labels[p]=l 
        replace_labels = replace_sign

        attention_mask = [1]* tokens_length + [0] * pad_length

        ####change list to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        mask_labels = torch.tensor(mask_labels, dtype=torch.long)
        replace_labels = torch.tensor(replace_labels, dtype=torch.long)
        # print(input_ids, attention_mask, mask_labels, replace_labels)
        # bb
        return input_ids, attention_mask, mask_labels, replace_labels,origin_tokens_length


    def process_description(self, sent):

        sent_tokens = self.tokenizer.tokenize(self.pre_caption(sent))
        tokenizer_vocab = list(self.tokenizer.vocab.keys())

        tokens_length = len(sent_tokens)
        mask_token = []
        mask_sign = [0]*tokens_length
        replace_token = []
        replace_sign = [0]*tokens_length
        
        pos_index = list(range(tokens_length))
        random.shuffle(pos_index)
        tokenizer_vocab = ['[MASK]'] if tokenizer_vocab is None else tokenizer_vocab

        #### mask sentence
        mask_length = int(tokens_length * 0.15 * 0.8)
        for i in range(mask_length):
            mask_token.append(sent_tokens[pos_index[i]])
            mask_sign[pos_index[i]] = 1
            sent_tokens[pos_index[i]] = '[MASK]'

        replace_length = int(tokens_length * 0.15 * 0.1)
        for i in range(mask_length, mask_length+replace_length):
            mask_token.append(sent_tokens[pos_index[i]])
            mask_sign[pos_index[i]] = 1

            replace_token.append(sent_tokens[pos_index[i]])
            replace_sign[pos_index[i]] = 1
            if random.random() < 0.5:
                sent_tokens[pos_index[i]] = random.choice(tokenizer_vocab)
            else:
                sent_tokens[pos_index[i]] = self.search_antonym(sent_tokens[pos_index[i]])
        added_tokens = self.tokenizer.tokenize('the image description is')
        sent_tokens = added_tokens + sent_tokens
        replace_sign = [0]* (len(added_tokens)) + replace_sign
        mask_sign = [0] * (len(added_tokens)) + mask_sign


        return sent_tokens, mask_token, mask_sign ,replace_token, replace_sign


    def search_synonym(self, word, label=None):
        '''
        if finded return syn else return word
        '''
        assert label in ('n','a','v',None)
        syns = wn.synsets(word)
        syns_set = []
        for syn in syns:
            syns_set.extend(syn.lemma_names())
        syns_set = set(syns_set)
        if syns_set:
            word = random.choice(list(syns_set))

        return word


    def search_antonym(self, word, label=None):
        anto = []
        for syn in wn.synsets(word):
            for lm in syn.lemmas():
                if lm.antonyms():
                    anto.append(lm.antonyms()[0].name())
        return random.choice(anto) if anto else word
            

    def pre_caption(self, caption,max_words=None):
        caption = re.sub(
            # r"([,.'!?\"()*#:;~])",
            r"([,.'!\"()*#:;~])",
            '',
            caption.lower(),
        ).replace('-', ' ').replace('/', ' ').replace('<person>', 'person').replace('<br>',' ')

        caption = re.sub(
            r"\s{2,}",
            ' ',
            caption,
        )
        caption = caption.rstrip('\n') 
        caption = caption.strip(' ')

        #truncate caption
        if max_words is not None:
            caption_words = caption.split(' ')
            if len(caption_words)>max_words:
                caption = ' '.join(caption_words[:max_words])
                
        return caption
