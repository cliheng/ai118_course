from modelscope import AutoTokenizer, AutoModel
import torch

def load_embeddings(sentences):
    # 加载模型和分词器
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
    model.eval()

    # 编码输入
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # 获取嵌入向量
    with torch.no_grad():
        model_output = model(**encoded_input)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    
    # normalize embeddings
    sentence_embeddings = sentence_embeddings.numpy().tolist()
    
    return sentence_embeddings

if __name__ == "__main__":
    sentences = ["样式数据-1"]
    sentence_embeddings = load_embeddings(sentences)
    print("Sentence embeddings:", sentence_embeddings)