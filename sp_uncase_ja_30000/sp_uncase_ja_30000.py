# coding: utf-8

"""
A tokenizer for Japanese texts via sentencepiece

- the coressponding model name is sp_uncase_ja_30000
- the sentencepiece model fitted with jawiki
- uncased 30000 tokens (16 reserved tokens)
- a wrapper class used in BERT repository, FullTokenizer, was adapted.
"""


import re
import unicodedata
import sentencepiece as sp


PATH_PREFIX = __file__.strip('.py')


class FullTokenizer(object):
    
    TOKEN_UNK='<unk>'
    TOKEN_SPACE='\u2581'
    
    @staticmethod
    def load_dict(file_path):
        """
        Assumes a tab splitted file where each vocab is the first column of rows.
        Vocab indexes were decided with the order of apperance.
        return:
            vocab: token dict -> idx, inv_vocab: idx -> token dict
        """
        
        # We don't use csv library to read .vocab but read line by line
        # because sp don't escape special characters when outputing it
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [_ for _ in f.read().split('\n') if len(_) > 0]
        vocab_list = [_.split('\t')[0] for _ in lines]
        
        vocab = {v:i for i, v in enumerate(vocab_list)}
        inv_vocab = {i:v for i, v in enumerate(vocab_list)}
        
        return vocab, inv_vocab
    
    def __init__(self, model_file=None, vocab_file=None, mapping=None, **kwargs):
        """
        initializes a tokenizer
        args:
            model_file: path to a .model file
            vocab_file: path to a .vocab file
        """
        
        if 'do_lower_case' in kwargs:
            raise ValueError('This tokenizer supports uncased text only.')
        
        self.prefix = PATH_PREFIX
        
        model_file = model_file or (self.prefix + '.model')
        vocab_file = vocab_file or (self.prefix + '.vocab')
        
        self.tokenizer = SentencePieceTokenizer(model_file)
        self.vocab, self.inv_vocab = self.load_dict(vocab_file)
        self.unk_id = self.vocab[self.TOKEN_UNK]
        
        if mapping is not None:
            self.rewrite_dict(mapping)
    
    def tokenize(self, text, as_ids=False, remove_spaces=False):
        tokens = self.tokenizer.tokenize(text)
        if remove_spaces:
            tokens = list(filter(lambda t: t!=self.TOKEN_SPACE, tokens))
        if as_ids:
            return self.convert_tokens_to_ids(tokens)
        return tokens
    
    def __call__(self, *args, **kwargs):
        return self.tokenize(*args, **kwargs)
    
    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.get(_, self.unk_id)for _ in tokens]

    def convert_ids_to_tokens(self, ids):
        return [self.inv_vocab.get(_, self.TOKEN_UNK)for _ in ids]
    
    def rewrite_dict(self, mapping):
        """
        Rewrites vocab and inv_vocab dict accroding to mapping.
        This don't affect the sentencepeice model.
        For example, after you rewriting 'unused_0' to '[CLS]',
        convert_tokens_to_ids('[CLS]') will return [3] (the id that unsed_0 was mapped)
        but tokenize('[CLS]abc[$]') will not return ['[CLS]', ...].
        args:
            mapping: a dict {'before_rewrite_vocab':'after_vocab', ...}
        """
        cannot_change = ['<pad>', '<unk>', '\u2581']
        
        for k, v in mapping.items():
            if k in cannot_change:
                raise ValueError('vocab %s cannot be changed.'%(k))
            if not (k in self.vocab):
                raise ValueError('vocab %s does not exists in current dict.'%(k))
            if v in self.vocab:
                raise ValueError('vocab %s already exists.'%(v))
        
        for k, v in mapping.items():
            i = self.vocab[v] = self.vocab.pop(k)
            self.inv_vocab[i] = v
            
    def summary(self):
        print('path_prefix=%s'%(self.path_prefix))
        print('num_of_vocab=%d'%(len(self.vocab)))


class SentencePieceTokenizer(object):
    
    nmt_norm_map = str.maketrans({
            # SPACES
            '\u0009':'\u0020',  # TAB
            '\u000A':'\u0020',  # LINE FEED
            '\u000C':'\u0020',  # FORM FEED
            '\u000D':'\u0020',  # CARRIAGE RETURN
            '\u1680':'\u0020',  # OGHAM SPACE MARK
            '\u200B':'\u0020',  # ZERO WIDTH SPACE
            '\u200E':'\u0020',  # LEFT-TO-RIGHT MARK
            '\u200F':'\u0020',  # RIGHT-TO-LEFT MARK
            '\u2028':'\u0020',  # LINE SEPARATOR
            '\u2029':'\u0020',  # PARAGRAPH SEPARATOR
            '\u2581':'\u0020',  # LOWER ONE EIGHT BLOCK
            '\uFEFF':'\u0020',  # ZERO WIDTH NO-BREAK
            '\uFFFD':'\u0020',  # REPLACEMENT CHARACTER
            '\u200C':'\u0020',  # ZERO WIDTH NON-JOINER
            '\u200D':'\u0020',  # ZERO WIDTH JOINER
            
            # Ascii Control characters
            '\u0001':'',
            '\u0002':'',
            '\u0003':'',
            '\u0004':'',
            '\u0005':'',
            '\u0006':'',
            '\u0007':'',
            '\u0008':'',
            '\u000B':'',
            '\u000E':'',
            '\u000F':'',
            '\u0010':'',
            '\u0011':'',
            '\u0012':'',
            '\u0013':'',
            '\u0014':'',
            '\u0015':'',
            '\u0016':'',
            '\u0017':'',
            '\u0018':'',
            '\u0019':'',
            '\u001A':'',
            '\u001B':'',
            '\u001C':'',
            '\u001D':'',
            '\u001E':'',
            '\u001F':'',
            
            #  <control-007F>..<control-009F>
            '\u007F':'',
            '\u008F':'',
            '\u009F':'',    
        })
    
    @classmethod
    def normalize_with_nmt_NFKC(
            cls,
            text, 
            treat_whitespace_as_suffix_=False, 
            add_dummy_prefix=True, 
            remove_extra_whitespaces=True, 
            escape_whitespaces=True,
            do_lower_case=True,
        ):
        """
        An emulation of sp normalizer with nmt NFKC.
        This method is not required before inputing tokens into the sp tokenizer because it normalize them by itself.
        You can know in advance what the whole string of the tokenized text will be.
        """
        
        # custom mapping for nmt
        text = text.translate(cls.nmt_norm_map)
        # tilde protection (storing)
        tildes = filter(lambda c: c == '\uFF5E' or c == '\u007E', text)
        # nfkc normalization
        text = unicodedata.normalize('NFKC', text)
        # tilde protection (restoring)
        text = ''.join([c if c != '\u007E' else next(tildes) for c in text])
        
        # triming extra spaces
        if remove_extra_whitespaces:
            text = re.sub('\u0020+', '\u0020', text.strip())
        # dummy space
        if add_dummy_prefix:
            if treat_whitespace_as_suffix_:
                text = text + '\u0020'
            else:
                text = '\u0020' + text
        # escaping spaces
        if escape_whitespaces:
            text = text.replace('\u0020', '\u2581')
        
        # do_lower_case which is a part of BERT script
        if do_lower_case:
            text = text.lower()
        return text
    
    def __init__(self, model_file=None, enabled_sampling=False):
        
        self.tokenizer = sp.SentencePieceProcessor()
        if not self.tokenizer.Load(model_file):
            raise Exception('Failed to load a model you specified: %s'%(model_file))
        
        self.enabled_sampling = enabled_sampling
        self.sampling_nbest_size = -1
        self.sampling_alpha = 0.1
    
    def tokenize(self, text):
        
        text = text.lower()
        
        if self.enabled_sampling:
            output_tokens = self.tokenizer.SampleEncodeAsPieces(
                    text, 
                    nbest_size=self.sampling_nbest_size, 
                    alpha=self.sampling_alpha,
                )
        else:
            output_tokens = self.tokenizer.EncodeAsPieces(text)
        
        return output_tokens

