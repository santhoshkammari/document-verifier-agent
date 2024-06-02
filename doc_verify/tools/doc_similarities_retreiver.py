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



def retreive_document_information(document,questions:List):
    path = "/home/ntlpt59/MAIN/LLM/lang-graph-explore/doc_verify/session"
    doc_scores = []
    alldocs = os.listdir(path)
    for doc in alldocs:
        doc = doc.split(".")[0]
        doc_scores.append(sentences_similarity_score(doc,document))

    document = alldocs[doc_scores.index(max(doc_scores))].split(".")[0]
    print(f"document: {document}")
    k = 4
    with open(f'/home/ntlpt59/MAIN/LLM/lang-graph-explore/doc_verify/session/{document}.txt',"r") as f:
        text = f.read()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=25,
                                                   chunk_overlap=10)
    splits = text_splitter.split_text(text)
    matches = []
    for split in splits:
        for question in questions:
            question = question.get("question") if isinstance(question, dict) else question
            score = sentences_similarity_score(split, question)
            matches.append((score,question,split))
    sorted_matches = sorted(matches,key=lambda x:x[0],reverse=True)
    return [_[-1] for _ in sorted_matches[:k]]

if __name__ == '__main__':
    print(retreive_document_information("bol",["what is the date of the bill of lading?"]))
