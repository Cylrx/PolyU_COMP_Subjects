# Notes

Since someone requested the codebase for this project, I’ve decided to upload the available source code here.  
Unfortunately, due to the time that has passed, many of the original files **have been lost** (e.g., `requirements.txt`, visualization scripts). These are the only files I was able to recover from my WeChat history.

Apologies for any reproducibility issues this may cause, and thanks for your understanding.

---

# Limitations

Looking back at this codebase nearly one year later, I can see several shortcomings that mostly reflect my limited understanding of ML at the time. I’m sharing them here in case they **spark ideas** or improvements for your own work:

1. Both the reduced bottleneck and the linear decoder projection seem to improve separability, but why? Is the linear projection actually necessary, or was my original hypothesis about the underlying issue incorrect?
2. Is an autoencoder even needed in the first place? Could we instead compute cosine similarity directly between high-dimensional word embeddings and then apply KNN?