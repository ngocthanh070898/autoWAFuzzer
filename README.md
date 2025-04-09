### OVERVIEW
**Requirement:**

```
!pip install langchain
!pip install -U langchain-community
!pip install --upgrade trl
# !pip install transformers==4.35.0
!pip install accelerate
!pip install -i https://test.pypi.org/simple/ bitsandbytes
!pip install --upgrade bitsandbytes
!pip install --upgrade accelerate
!pip install peft==0.10.0
!pip install transformers==4.37.2
!pip install chromadb

!pip install pymisp
```

### RUN CODE WITH CODELLAMA

**Pretraining:**

```
!python /content/drive/MyDrive/NCKH/Inference/pretrain_codellama.py \
    --data_path "/content/drive/MyDrive/NCKH/Datasets/XSS/small_XSS_Dataset.txt" \
    --output_dir "/content/drive/MyDrive/NCKH/results/trained_codellama" \
    --model_name "/content/drive/MyDrive/NCKH/CodeLlama-7b-hf" \
    --epochs 1 \
    --batch_size 1 \
    --use_8bit
```

