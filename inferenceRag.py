import torch
from tqdm import tqdm
import os
import pandas as pd
from transformers import GPT2Tokenizer
from gpt2 import GPT2HeadWithValueModel, respond_to_batch
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
import datetime
import argparse

tqdm.pandas()
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

parser = argparse.ArgumentParser(description="")
parser.add_argument('--lm_name')
parser.add_argument('--ref_lm_name')
parser.add_argument('--total_nums')
parser.add_argument('--txt_in_len')
parser.add_argument('--txt_out_len')
parser.add_argument('--savePath')
parser.add_argument('--vector_db_path')
args = parser.parse_args()

lm_name = args.lm_name
ref_lm_name = args.ref_lm_name
total_nums = int(args.total_nums)
txt_in_len = int(args.txt_in_len)
txt_out_len = int(args.txt_out_len)
savePath = args.savePath
vector_db_path = args.vector_db_path

# Load tokenizer & models
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2HeadWithValueModel.from_pretrained(lm_name)
gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained(ref_lm_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpt2_model.to(device).eval()
gpt2_model_ref.to(device).eval()

# Load ChromaDB
embedding = HuggingFaceEmbeddings(model_name="/content/drive/MyDrive/NCKH/all-MiniLM-L6-v2")
vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embedding)

def retrieve_relevant_payloads(query, top_k=5):
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.get_relevant_documents(query)
    return [doc.page_content for doc in retrieved_docs]

bs = 64
wafData = pd.DataFrame()
wafData['content'] = ['0' for _ in range(30000)]
wafData['tokens'] = wafData['content'].progress_apply(lambda x: tokenizer.encode(x, return_tensors="pt").to(device)[0, :txt_in_len])
wafData['query'] = wafData['tokens'].progress_apply(lambda x: tokenizer.decode(x))

responseList = []
starttime = datetime.datetime.now()
while len(responseList) < total_nums:
    torch.cuda.empty_cache()
    df_batch = wafData.sample(bs)
    query_tensors = torch.stack(df_batch['tokens'].tolist())
    
    # Lấy các payload liên quan từ VectorDB
    retrieved_payloads = [retrieve_relevant_payloads(tokenizer.decode(qt), top_k=5) for qt in query_tensors]
    retrieved_texts = [" ".join(payloads) for payloads in retrieved_payloads]
    
    # Cập nhật input cho GPT-2
    augmented_queries = [qt + " " + rt for qt, rt in zip(df_batch['query'], retrieved_texts)]
    input_tensors = torch.stack([tokenizer.encode(q, return_tensors="pt").to(device)[0, :txt_in_len] for q in augmented_queries])
    
    response_tensors = respond_to_batch(gpt2_model, gpt2_model_ref, input_tensors, txt_len=txt_out_len)
    responseList += [tokenizer.decode(response_tensors[i, :]).split('!')[0] for i in range(bs)]

endtime = datetime.datetime.now()
print((endtime - starttime).seconds)

df_results = pd.DataFrame()
df_results['response'] = responseList
df_results['query'] = '0'
df_results['data'] = df_results['query'] + df_results['response']
df_results[['data']].to_csv(savePath, index=False)


