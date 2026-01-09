import math
import pandas as pd
import numpy as np
import ollama
from ollama import EmbeddingsResponse
from tqdm import tqdm

# Chunk-wise embedding generation

class LLMEmbedding:
    def __init__(self, model_id, prompt):
        self.model_id = model_id
        self.embed_dim = len(ollama.embeddings(model=model_id, prompt='test')['embedding'])
        self.prompt = prompt

    def get(self, text) -> list:
        message = f'{self.prompt}:\n"{text}"'
        response: EmbeddingsResponse = ollama.embeddings(model=self.model_id, prompt=message)
        return response['embedding']

    def get_df(self, df) -> pd.Series:
        return df['headline_text'].apply(self.get)

    def get_np(self, df) -> np.ndarray:
        return self.get_df(df).to_numpy()

    def size(self):
        return self.embed_dim


CHUNK_SIZE = 10000
SAMPLE_SIZE = 0
SEED = 42

def main():
    input_file = '../abcnews-date-text-cleaned.csv'
    output_file = 'abcnews-date-text-embed-nomic.csv'
    model_id = "nomic-embed-text"

    # LLM's next token vector is used as prediction.
    # Therefore, we frame the prompt to guide the model towards generating 
    # embeddings suitable for semantic clustering.
    prompt = """
    You are an expert in news recommendation systems. 
    Please classify the following news headline with a single word or a short phrase that captures its theme
    """ 
    skiprows = None

    df = pd.read_csv(input_file)
    size = df.shape[0]

    if SAMPLE_SIZE > 0:
        sampled_df = df.sample(n=SAMPLE_SIZE, random_state=SEED)
        skiprows = list(set(range(len(df))) - set(sampled_df.index))
        total_rows = SAMPLE_SIZE
    else:
        total_rows = size

    del df  # Free up memory
    print(f'Total rows to process: {total_rows}')

    # Calculate the total number of chunks
    total_chunks = math.ceil(total_rows / CHUNK_SIZE)

    embeddings_model = LLMEmbedding(model_id, prompt)
    embedding_columns = [f'embedding_{i+1}' for i in range(embeddings_model.size())]
    print(f'Embedding Dimension: {embeddings_model.embed_dim}')

    write_mode = 'w'
    header_written = False  # To ensure headers are written only once

    with pd.read_csv(input_file, chunksize=CHUNK_SIZE, skiprows=skiprows) as reader:
        for chunk in tqdm(reader, total=total_chunks, desc="Processing Chunks"):
            embeddings = embeddings_model.get_df(chunk)
            embeddings_df = pd.DataFrame(embeddings.tolist(), columns=embedding_columns)

            combined_df = pd.concat([chunk.reset_index(drop=True), embeddings_df], axis=1)

            combined_df.to_csv(
                output_file,
                index=False,
                mode=write_mode,
                header=not header_written
            )

            if not header_written:
                header_written = True
                write_mode = 'a'

    print("All Done.")

if __name__ == '__main__':
    main()