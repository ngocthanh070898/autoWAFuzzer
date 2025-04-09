import torch
from tqdm import tqdm
import os
import pandas as pd
tqdm.pandas()
from transformers import GPT2Tokenizer, LlamaTokenizer, LlamaForCausalLM
from transformers import CodeLlamaTokenizer
from gpt2 import GPT2HeadWithValueModel, respond_to_batch 
import datetime
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = '6'

parser = argparse.ArgumentParser(description="")
parser.add_argument('--lm_name')
parser.add_argument('--ref_lm_name')
parser.add_argument('--total_nums')
parser.add_argument('--txt_in_len')
parser.add_argument('--txt_out_len')
parser.add_argument('--savePath')
args = parser.parse_args()

lm_name = args.lm_name
ref_lm_name = args.ref_lm_name
total_nums = int(args.total_nums)
txt_in_len = int(args.txt_in_len)
txt_out_len = int(args.txt_out_len)
savePath = args.savePath

# Load GPT-2 tokenizer và model fine-tuned với RL
tokenizer = gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2HeadWithValueModel.from_pretrained(lm_name)

# Load trained CodeLLaMA model làm reference model
#llama_tokenizer = LlamaTokenizer.from_pretrained(ref_lm_name)
#codellama_model = LlamaForCausalLM.from_pretrained(ref_lm_name)
llama_tokenizer = CodeLlamaTokenizer.from_pretrained(ref_lm_name, legacy=False)
#codellama_model = LlamaForCausalLM.from_pretrained(ref_lm_name, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)

from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Hoặc thử load_in_4bit=True nếu vẫn lỗi
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

codellama_model = LlamaForCausalLM.from_pretrained(
    ref_lm_name,
    quantization_config=bnb_config,
    device_map="auto"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_ = gpt2_model.eval().to(device)
_ = codellama_model.eval().to(device)

bs = 64
wafData = pd.DataFrame()
wafData['content'] = ['0' for _ in range(30000)]
wafData['tokens'] = wafData['content'].progress_apply(lambda x: gpt2_tokenizer.encode(x, return_tensors="pt").to(device)[0, :txt_in_len])
wafData['query'] = wafData['tokens'].progress_apply(lambda x: gpt2_tokenizer.decode(x))

responseList = []

starttime = datetime.datetime.now()
while len(responseList) < total_nums:
    torch.cuda.empty_cache()
    df_batch = wafData.sample(bs)
    query_tensors = torch.stack(df_batch['tokens'].tolist())

    # Gọi inference với mô hình GPT-2 finetune và CodeLLaMA reference
    response_tensors = respond_to_batch(gpt2_model, codellama_model, query_tensors, txt_len=txt_out_len)

    responseList += [gpt2_tokenizer.decode(response_tensors[i, :]).split('!')[0] for i in range(bs)]

endtime = datetime.datetime.now()
print((endtime - starttime).seconds)

df_results = pd.DataFrame()
df_results['response'] = responseList
df_results['query'] = '0'
df_results['data'] = df_results['query'] + df_results['response']
df_results[['data']].to_csv(savePath, index=False)
