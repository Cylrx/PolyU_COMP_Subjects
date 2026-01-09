# Get Started

## Basic Install
This project uses uv as its package + venv manager. 
If you haven't already, consider installing it by simply running one line of command (see https://docs.astral.sh/getting-started/installation/)

Once you've installed uv, navigate to project root, and ensure `pyproject.toml` can be found. 

```
cd ./HeartAttackAnalysis
uv install
```

Then run the project via: 

```
uv run python main.py
```
And you are set to go!

---

## Installing LimiX

The project uses TabPFNv2.5 as its default backbone. For absolute State-of-the-Art performance, consider installing LimiX.
To use LimiX, follow the instructions below. Alternative, disable LimiX in config and fallback to using TabPFNv2.5

```
wget -O flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

uv python pin 3.12.7
uv add torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1
uv add flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
uv add scikit-learn  einops  huggingface-hub matplotlib networkx numpy pandas  scipy tqdm typing_extensions xgboost kditransform hyperopt

# NOTE: ensure that you are in project root.
git clone https://github.com/limix-ldm/LimiX.git
```

## LLM As a Judge

To run LLM judge in the mining stage, you need to obtain your own OpenAI and Gemini API Keys.
Then run the commands: 

```
OPENAI_API_KEY=<your-openai-api-key>
export GOOGLE_API_KEY=<your-google-api-key>
```