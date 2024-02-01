from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# semanticSimilarity will be calculation the similarity score and do the preprocessing
class semanticSimilarity():
    
    model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    def __init__(self, sent1, sent2):
        self.model = semanticSimilarity.model
        self.sent1 = sent1
        self.sent2 = sent2
        
    # Mean Pooling - Take attention mask into account for correct averagi
    def __mean_pooling(self, model_output, attention_mask):
        
        token_embeddings = model_output[0] # First element of the model contains all the token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        tensorss =  torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min = 1e-9)
        return tensorss
    
    def __tokenize(self):
        
        sentences = [self.sent1, self.sent2]
        tokenizer = AutoTokenizer.from_pretrained(self.model)
        model = AutoModel.from_pretrained(self.model)
        
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        
        with torch.no_grad():
            model_output = model(**encoded_input)
            
        sentence_embeddings = self.__mean_pooling(model_output, encoded_input['attention_mask'])
        
        return sentence_embeddings
    
    def score_calculations(self):
        
        embeddings = self.__tokenize()
        similarity_matrix = cosine_similarity(embeddings)
        
        similarity_score = similarity_matrix[0,1]
        
        return round(abs(similarity_score), 2)
        