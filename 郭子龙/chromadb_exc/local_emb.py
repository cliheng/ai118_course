from modelscope import AutoTokenizer, AutoModel
import torch
def local_embdding(sentences):
    #tokenizer输入文本转换模型输入需要变量类型
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
    model.eval()
# Sentences we want sentence embeddings for

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
    #生成embedding
    with torch.no_grad():
        model_output = model(**encoded_input)
        #从结果中抽取模型生成embedding
        sentence_embeddings = model_output[0][:, 0]

    sentence_embeddings = sentence_embeddings.numpy().tolist()
    #print('Sentence embddings:',len(sentence_embeddings))
    return sentence_embeddings
if __name__ == "__main__":
 
    sentences = ["样例数据-1"]

    sentence_embeddings = local_embdding(sentences)
print(sentence_embeddings)