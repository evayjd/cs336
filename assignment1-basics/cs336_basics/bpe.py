import regex as re
import collections
from collections import defaultdict
from tqdm.contrib.concurrent import process_map

PAT=re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def read_file(input_path):
    with open(inpput_path,'r',encoding='utf-8') as f:
        return f.read()

def split_by_special(text,special_token):
    if not special_token:
        return [text]
    special_tokens=sorted(special_tokens,key=len,reverse=True)# To avoid prefix-collision issue
    pattern=re.compile("|".join(re.escape(token) for token in special_tokens))
    chunks=pattern.split(text)
    result = []
    for c in chunks:
        if c:
            result.append(c)

    return result# remove empty strings

def count_word_parallel(text):
    word_cnt=defaultdict(int)
    for m in PAT.finditer(text):
        word=m.group(0)
        word_bytes=tuple(bytes([b]) for b in word.encode('utf-8'))
        word_cnt[word_bytes]+=1
    return word_cnt

def train_bpe(input_path,vocab_size,special_tokens):
    text=read_file(input_path)
    chunks=split_by_special(text,special_tokens)
    if len(chunks)<4:
        word_dicts=[count_word_parallel(chunk) for chunk in chunks]
    else:
        word_dicts=process_map(count_word_parallel,chunks,chunksize=1)
        
    word_counts=defaultdict(int)
    for word_dict in word_dicts:
        for word, count in word_dict.items():
            word_counts[word]+=count    
            
    pair_counts=collections.Counter()
    pair_to_words=defaultdict(set)
    
    for word_tuple,count in word_counts.items():
        for i in range(len(word_tuple)-1):
            pair=(word_tuple[i],word_tuple[i+1])
            pair_counts[pair]+=count
            pair_to_words[pair].add(word_tuple)
            
    merge=[]
    vocab={i:bytes([i]) for i in range(256)}
    current_vocab_size=256
    
    n_merges=vocab_size-256-len(special_tokens)
    for i in range(n_merges):
        if not pair_counts:
            break
        best_pair=max(pair_counts.keys(),key=lambda p:(pair_counts[p],p))
        if pair_counts[best_pair]==0:
            break
        
        new_token=best_pair[0]+best_pair[1]
        merges.append(best_pair)
        
        vocab[current_vocab_size]=new_token
        current_vocab_size+=1
        
        
        word_to_update=list(pair_to_words[best_pair])
        for word_tuple in words_to_update:
            count=word_counts[word_tuple]
            
            for j in range(len(word_tuple)-1):
                p=(word_tuple[j],word_tuple[j+1])
                pair_counts[p]-=count
                pair_to_words[p].remove(word_tuple)
            
            new_word_tupe=[]
            skip=False
            for j in range(len(word_tuple)):
                if skip:
                    skip=False
                    continue
                if j<len(word_tuple)-1 and (word_tuple[j],word_tuple[j+1])==best_pair:
                    new_word_tuple.append(new_token)
                    skip=True
                else:
                    new_word_tuple.append(word_tuple[j])
            new_word_tuple=tuple(new_word_tuple)
            
            del word_counts[word_tuple]
            word_counts[new_word_tuple]=count
            for j in range(len(new_word_tuple)-1):
                p=(new_word_tuple[j],new_word_tuple[j+1])
                pair_counts[p]+=count
                pair_to_words[p].add(new_word_tuple)
            
            for special in special_tokens:
                vocab[current_vocab_size]=special.encode('utf-8')
                current_vocab_size+=1
                
            return vocab,merge