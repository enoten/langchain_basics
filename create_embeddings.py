import warnings
from sentence_transformers import SentenceTransformer
warnings.filterwarnings("ignore")

phrase_model = SentenceTransformer("all-mpnet-base-v2")

input_text = "Modify my access profile to include admin rights"
embeds = phrase_model.encode(input_text)

print()
print("Embedding for the input text:")
print()
print(input_text)
print()
print(embeds)
print(len(embeds))