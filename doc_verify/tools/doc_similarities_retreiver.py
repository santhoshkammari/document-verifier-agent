import json
import os
from typing import List

import torch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
llm = SentenceTransformer(model_name_or_path="all-MiniLM-L6-v2")


def sentences_similarity_score(split, question):
    enc1 = llm.encode(split)
    enc2 = llm.encode(question)
    return F.cosine_similarity(torch.tensor(enc1),torch.tensor(enc2),dim=0).item()



def retreive_document_information(document,questions:List,statement):
    questions.append({"variation":statement}) ### add original question along with variations
    path = "session"
    doc_scores = []
    alldocs = os.listdir(path)
    for doc in alldocs:
        doc = doc.split(".")[0]
        doc_scores.append(sentences_similarity_score(doc,document))

    document = alldocs[doc_scores.index(max(doc_scores))]
    print(f"document: {document}")
    k = 25
    with open(f'session/{document}', "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    print(f'{len(text)=}')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50,
                                                   chunk_overlap=10)
    splits = text_splitter.split_text(text)
    matches = []
    with open(f"session/{document} textsplit.json",'w') as f:
        json.dump(splits,f,indent=2)

    print(f"splits saved for {document}")

    for split in splits:
        for question in questions:
            question = question.get("variation") if isinstance(question, dict) else question
            score = sentences_similarity_score(split, question)
            matches.append((score,question,split))


    sorted_matches = sorted(matches,key=lambda x:x[0],reverse=True)

    with open(f"session/{document} matches.json",'w') as f:
        json.dump(sorted_matches,f,indent=2)

    split_filterd = [_[-1] for _ in sorted_matches]
    seen = set()
    unique_splits = []
    for s in split_filterd:
        if s not in seen:
            unique_splits.append(s)
            seen.add(s)

    return unique_splits[:k]

if __name__ == '__main__':
    print(retreive_document_information("bol",["what is the date of the bill of lading?"]))
