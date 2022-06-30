from dataUtil import get_seg_features

def load_sentences(path):
    sentences=[]
    sentence=[]
    for line in  open(path,encoding='utf-8'):
        line=line.rstrip()
        if len(line)==0:
            sentences.append(sentence)
            sentence=[]
        else:
            line=line.split()
            sentence.append(line)

    return sentences

def updata_tag_scheme(sentences,tag_scheme):
    if tag_scheme=='bio':
        return
    if tag_scheme=='bioes':
        for i in range(len(sentences)):
            sentence=sentences[i]
            for j in range(len(sentence)):
                word=sentence[j]
                if word[-1][0]=='B':
                    if j+1!=len(sentence) and sentence[j+1][-1][0]=='I':
                        continue
                    else:
                        word[-1]=word[-1].replace('B-','S-')
                elif word[-1][0]=='I':
                    if j+1!=len(sentence) and sentence[j+1][-1][0]=='I':
                        continue
                    else:
                        word[-1]=word[-1].replace('I-','E-')

def char_mapping(sentences):
    create_dico={}
    for sentence in sentences:
        for word in sentence:
            if word[0] in create_dico:
                create_dico[word[0]]+=1
            else:
                create_dico[word[0]]=1

    create_dico['<UNK>']=1000000
    create_dico['<PAD>'] = 1000001
    sorted_items=sorted(create_dico.items(),key=lambda x:-x[-1])
    char2id={item[0]:i for i,item in enumerate(sorted_items)}
    id2char={v:k for k,v in char2id.items()}
    return create_dico,char2id,id2char

def tag_mapping(sentences):
    dico={}
    for sentence in sentences:
        for word in sentence:
            if word[-1] in dico:
                dico[word[-1]]+=1
            else:
                dico[word[-1]]=1

    sorted_items=sorted(dico.items(),key=lambda x:-x[1])
    tag2id={v[0]:i for i,v in enumerate(sorted_items)}
    id2tag={v:k for k,v in tag2id.items()}
    return tag2id,id2tag

def prepare_dataset(sentences,char2id,tag2id):
    data=[]
    for sentence in sentences:
        string=[word[0] for word in sentence]
        chars=[char2id[w if w in char2id else '<UNK>'] for w in string]
        tags=[tag2id[word[-1]] for word in sentence]
        segs=get_seg_features("".join(string))

        data.append([string,chars,segs,tags])

    return data

if __name__=='__main__':
    sentences = load_sentences('data/example.dev')
    updata_tag_scheme(sentences,'bioes')
    tag_mapping(sentences)