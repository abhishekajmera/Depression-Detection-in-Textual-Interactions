import os
import logging
from allennlp.common.from_params import T
import torch
from typing import Dict, List, Iterable, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from overrides import overrides
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import pandas as pd
import contractions,re,itertools
import random
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField, ListField, ArrayField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer

from utility import separate_train_dev_test, original_separation
from constant import F_EMO, F_ACT, F_TOPIC, F_TEXT, DAIC_DIR, ERISK_DIR, DAIC_TRAIN_IDX, DAIC_DEV_IDX, DAIC_TEST_IDX, DAIC_TRAIN_PHQ9_BI, DAIC_DEV_PHQ9_BI, DAIC_TEST_PHQ9_BI

logger = logging.getLogger(__name__)

@DatasetReader.register("dialog_reader")
class DialogReader(DatasetReader):
    """
    Reads a file of DAIC / DD conversations
         
    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``SpacyTokenizer()``)
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_sequence_length: int = None,
        params: Dict = None
    ) -> None:

        super(DialogReader, self).__init__()
        self._tokenizer = tokenizer or SpacyWordSplitter()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._max_sequence_length = max_sequence_length
        self.encode_turns = params['encode_turns']
        # self.MODE = params['MODE']
        self.has_emo = params['has_emo']
        self.has_act = params['has_act']
        self.has_topic = params['has_topic']
        self.has_phq = params['has_phq']
        self.has_phqbi = params['has_phqbi']
        # self.daic_or_erisk = 'erisk'
        # multilingual model, similar embeddings as the bert-base-nli-stsb-mean-token model, trained on 50+ lang
        self.turn_encoder = SentenceTransformer('stsb-xlm-r-multilingual') if self.encode_turns else None
        if params['orig_separation']:
            self.idxtrain, self.idxdev, self.idxtest = original_separation()
        else:
            self.idxtrain, self.idxdev, self.idxtest = separate_train_dev_test()
        # initiate subsets
        self.train1, self.dev1, self.test1 = None, None, None
        self.train2, self.dev2, self.test2 = None, None, None
        if self.has_emo or self.has_act or self.has_topic:
            self.train1, self.dev1, self.test1 = self.load_dailydata()
        # if self.daic_or_erisk=="daic":
        self.train_daic, self.dev_daic, self.test2 = self.load_daicdata()   
        # elif self.daic_or_erisk=="erisk":
        self.train_erisk, self.dev_erisk = self.load_eriskdata()

        self.train2 = shuffle(self.train_daic + self.train_erisk)
        self.dev2 = shuffle(self.dev_daic + self.dev_erisk)
        
        # print("DEBUG1: train1 len, ", len(self.train1))
        # print("DEBUG1: train2 len, ", len(self.train2))

    def load_dailydata(self) -> List[T]:
        #parse emotion file, speech act file, topic file and text file 
        #return train, dev and test. each instance (1 dialog) is in a zipped list
        insts_train, insts_dev, insts_test = [], [], []

        in_topic = open(F_TOPIC, 'r', encoding="utf8")
        in_emo = open(F_EMO, 'r', encoding="utf8")
        in_act = open(F_ACT, 'r', encoding="utf8")
        in_dial = open(F_TEXT, 'r', encoding="utf8")

        for line_count, (line_dial, line_emo, line_act, line_topic) in enumerate(zip(in_dial, in_emo, in_act, in_topic)):
            inst = [[], [], [], []] #initiate an instance for each dialogue, sub-lists are text, speech act, emotion, topic

            seqs = line_dial.split('__eou__')
            seqs = seqs[:-1]

            emos = line_emo.split(' ') #string
            emos = emos[:-1]

            acts = line_act.split(' ') #string
            acts = acts[:-1]

            top = line_topic.strip() #string

            seq_len = len(seqs)
            emo_len = len(emos)
            act_len = len(acts)

            # print("DEBUG2: ", line_count)
            # print(acts)
            assert seq_len == emo_len == act_len, f"Line {line_count+1}: unmatched dialogue text ({seq_len}) & emotion ({emo_len}) & speech act ({act_len})!"

            inst[0] = [seq.strip() for seq in seqs]
            inst[1] = acts
            inst[2] = emos
            inst[3] = top

            if line_count+1 in self.idxtrain:
                insts_train.append(inst)
            elif line_count+1 in self.idxdev:
                insts_dev.append(inst)
            elif line_count+1 in self.idxtest:
                insts_test.append(inst)
        print(len(insts_train), len(insts_dev), len(insts_test))
        
        in_topic.close()
        in_emo.close()
        in_act.close()
        in_dial.close()
        return insts_train, insts_dev, insts_test

    def load_eriskdata(self) -> List[T]:
        
        def html_url(x):
            x=re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)','',x,flags=re.MULTILINE)
            s=BeautifulSoup(x,'html.parser')
            x=s.get_text()
            return x

        def contraction(x):
            x=contractions.fix(x)
            x=re.sub("[^a-zA-Z]+", " ", x)
            return x
        
        df = pd.read_csv(ERISK_DIR+'risk-golden-truth-test.txt', sep='\t', header=None, names=['col1', 'col2'])
        subject_labels=df['col1'].tolist()
        dep_labels=df['col2'].tolist()
        
        list_final = []
        for label, classify in zip(subject_labels, dep_labels):
            final_seq,final_user=[],[]
            for i in range(1,11):
                folder='chunk'+str(i)
                subject_name=label+'_'+str(i)
                final_folder=ERISK_DIR+folder+'/'+subject_name+'.xml'
                tree = ET.parse(final_folder) 
                root = tree.getroot() 
                seq,user=[],[]
                for w in root.findall('WRITING'):
                    texts=w.find('TEXT').text
                    #convert all reviews into lowercase.
                    texts=texts.lower()
                    #remove the HTML and URLs from the reviews
                    texts=html_url(texts)
                    #remove non-alphabetical characters
                    texts=re.sub("[^a-zA-Z']+", " ", texts)
                    #remove extra spaces
                    texts=re.sub(' +', ' ', texts)
                    #perform contractions on the review
                    texts=contraction(texts)
                    if texts==' ':
                        continue
                    seq.append(texts)
                    user.append(subject_name)
                if seq==[]:
                    continue
                list_final.append([seq,user,[classify]])
            #     final_seq.extend(seq)
            #     final_user.extend(user)
            # if final_seq==[]:
            #     continue
            # list_final.append([final_seq,final_user,[classify]])

        # train_idx = int(len(list_final)*0.6)
        # dev_idx = int(len(list_final)*0.8)
        # insts_train, insts_dev, insts_test = list_final[:train_idx], list_final[train_idx:dev_idx], list_final[dev_idx:]
        
        insts_train, insts_dev, _, _ = train_test_split(list_final, [i[2][0] for i in list_final], train_size=0.8, shuffle=True, stratify=None)
        # insts_train, insts_dev, _, _ = train_test_split(insts_train_dev, [i[2][0] for i in insts_train_dev], train_size=0.9, shuffle=True, stratify=None)

        print(len(insts_train), len(insts_dev))
        return insts_train, insts_dev

    def load_daicdata(self) -> List[T]:
        # read transcripts in data/daic/, follow original separation into train, dev and test, return 3 lists
        insts_train, insts_dev, insts_test = [], [], []
        DIR = DAIC_DIR
        for root, dirs, files in os.walk(DIR, topdown=True):
            for name in files:
                if os.path.isfile(os.path.join(root, name)) and name.endswith('.csv'):
                    daic_f = os.path.join(root, name)
                    f_idx = name.split('_')[0] 
                    if '-' in f_idx: #train or dev docs
                        f_idx = int(f_idx.split('-')[0])
                    else: #test docs
                        f_idx = int(f_idx)                      
                    with open(daic_f, 'r') as inf:
                        next(inf)
                        inst = [[], [], []] # an instance is a document contains 3 elements: [seq, ...], [spk, ...], [PHQ-9 binary, PHQ-9 multiclass]
                        lines = inf.readlines()
                        for l in lines:
                            l = l.strip()
                            if len(l.split('\t')) == 4:
                                _, _, spk, seq = l.split('\t')
                                inst[0].append(seq.lower())
                                inst[1].append(spk.lower())
                            else:
                                pass
                        assert len(inst[0]) == len(inst[1]), f'Unequql length of sequence {len(inst[0])} and speaker {len(inst[1])} in file {f_idx}.'
                        if f_idx in DAIC_TRAIN_IDX:
                            inst[2].append(DAIC_TRAIN_PHQ9_BI[DAIC_TRAIN_IDX.index(f_idx)])
                            insts_train.append(inst)
                        elif f_idx in DAIC_DEV_IDX:
                            inst[2].append(DAIC_DEV_PHQ9_BI[DAIC_DEV_IDX.index(f_idx)])
                            insts_dev.append(inst)
                        elif f_idx in DAIC_TEST_IDX:
                            inst[2].append(DAIC_TEST_PHQ9_BI[DAIC_TEST_IDX.index(f_idx)])
                            insts_test.append(inst)
        
        # print("DEBUG4: ", insts_test)
        print(len(insts_train), len(insts_dev), len(insts_test))
        return insts_train, insts_dev, insts_test


    def combine_daic_daily_inst(self, insts1, insts2):
        # print("DEBUG9: in combine datasets")
        # inst1=daily, inst2=daic
        # combine info from these 2 instances, return an newinst with form: [[seq],[spk],[emo_labels],[DA_labels],[topic_labels],[phq-9 scores]]
        combinsts = []
        if insts1: #[[seq], [act], [emo], [topic]]
            for i, inst in enumerate(insts1):
                newinst = [[], [], [], [], [], None] # no phq-0 score
                newinst[0] = inst[0]
                # randomly give spk labels for dailydialog since no spk names given in the corpus. every speech turn is a new spk. 
                newinst[1] = ['spk1', 'spk2'] * (len(inst[0])//2)
                newinst[2] = inst[2]
                newinst[3] = inst[1]
                newinst[4] = inst[3]
                if len(newinst[1]) < len(inst[0]):
                    newinst[1].append('spk1')
                combinsts.append(newinst)
        if insts2: #[[seq], [spk], [phq]]
            for i, inst in enumerate(insts2): #inst represents a doc
                newinst = [[], [], None, None, None, []] # no emo, da, topic labels
                newinst[0] = inst[0]
                newinst[1] = inst[1]
                newinst[5] = inst[2][0]
                combinsts.append(newinst)

        # print("DEBUG9: out of combine")
        return combinsts
        
    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        # obtain train/dev/test datasets obtained from @load_dailydata
        # take related dataset and pass to @text_to_instance
        # print("DEBUG10: enter",file_path)
        if file_path == 'train':
            subset = self.combine_daic_daily_inst(self.train1, self.train2)
        elif file_path == 'dev':
            subset = self.combine_daic_daily_inst(self.dev1, self.dev2)
        elif file_path == 'test':
            subset = self.combine_daic_daily_inst(self.test1, self.test2)  

        for i,inst in enumerate(subset): #[[seq],[spk],[emo_labels],[DA_labels],[topic_labels],[phq-9 scores]]
            #upzip the tuples
            inst_text, spks, label_emo, label_da, label_topic, label_phq = inst
            inst_text2 = [] #tokenized turns in inst_text
            for turn in inst_text:
                if self.turn_encoder:
                    inst_text2.append(turn)
                else:
                    inst_text2.append(self._tokenizer.tokenize(turn))

            print(f"{i} of {len(subset)}", end='\r')
            yield self.text_to_instance(inst_text2, label_emo, label_da, label_topic, label_phq)

        # print("DEBUG10: exits",file_path)


    @overrides
    def text_to_instance(
        self, 
        inst_text: Iterable,
        label_emo: Iterable,
        label_act: Iterable,
        label_topic: Iterable,
        label_phq: int
        ) -> Instance:
        
        fields: Dict[str, Field] = {}

        if self.turn_encoder:
            fields['lines'] = ListField([np.array(self.turn_encoder.encode(turn)) for turn in inst_text])
        else:
            fields['lines'] = ListField([TextField(t, self._token_indexers) for t in inst_text])
        
        if not label_emo is None:
            fields['label_emo'] = ListField([LabelField(int(emo), skip_indexing=True) for emo in label_emo]) #len=len(lines)
        else:
            fields['label_emo'] = ListField([LabelField(-1, skip_indexing=True)]*len(inst_text)) #len=len(lines)

        if not label_act is None: 
            fields['label_act'] = ListField([LabelField(int(act)-1, skip_indexing=True) for act in label_act]) #len=len(lines), @ATTETNION: -1 because speech act index starts with 1 and not 0
        else:
            fields['label_act'] = ListField([LabelField(-1, skip_indexing=True)]*len(inst_text))

        if not label_topic is None:
            fields['label_topic'] = LabelField(int(label_topic)-1, skip_indexing=True) #len=len(lines)
        else:
            fields['label_topic'] = LabelField(-1, skip_indexing=True) #len=len(lines)
        
        if not label_phq is None:
            fields['label_phq'] = LabelField(int(label_phq), skip_indexing=True)
        else:
            fields['label_phq'] = LabelField(-1, skip_indexing=True)

        return Instance(fields)
