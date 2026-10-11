# PoultryVision API

The backend for [PoultryVision](https://github.com/hilary-aoao/poultry-vision-frontend). It looks at a photo of a chicken dropping, then gives a result, a plain-language explanation and answers to follow-up questions. The explanations and answers come only from a verified knowledge base.

**Live API:** https://poultryvision-api.onrender.com (interactive docs at `/docs`)

## Endpoints

| Method | Path | What it does |
|---|---|---|
| GET | `/` | Health check |
| POST | `/predict` | Takes a photo, returns the result, the confidence, how reliable that class is, and the percentage for each class |
| POST | `/explain` | Takes a result, returns a short explanation, a reminder that this is not a vet's diagnosis, and suggested questions |
| POST | `/ask` | Takes a question about a result, returns an answer with its sources, or says it has no verified answer |

Example, asking a follow-up question:

```json
POST /ask
{"disease_class": "Coccidiosis", "question": "how long until my chicken gets better"}
```

```json
{
  "covered": true,
  "answer": "With prompt treatment, most birds show visible improvement within a day...",
  "sources": [{"id": "coccidiosis_2", "question": "How long does recovery from coccidiosis take?"}],
  "suggestions": ["Is it contagious?"]
}
```

## How the answers are kept honest

- **A knowledge base, not guesswork.** `knowledge_base.json` holds the facts for each disease, taken from veterinary and public health sources. The language model may only use those facts.
- **No local matching model.** The knowledge base is small, so for each question the entries for that one disease are put in the prompt. This also keeps the service light enough for a free server.
- **Answers must name their sources.** `/ask` asks the model to say which entries it used. If it names none, or an entry that does not exist, the answer is thrown away and the reply is "I don't have verified information on that".
- **Confidence is handled in code.** A result is "high" when the confidence is 85% or more and the class is reliable, "low" under 60%, and "moderate" otherwise. The explanation never repeats a percentage. Any sentence with a percentage, "the tool" or "the model" is removed before it reaches the farmer.
- **A second possible result is added by the code, in fixed words.** If another class is at 20% or more, the explanation ends with "There is a good, fair or small chance it could also be ...", chosen by how likely it is.
- **If the language model is down,** the explanation is built straight from the knowledge base.
- **Protection for the free tiers.** Each visitor can make 20 requests a minute, and all visitors share a budget of 24 language model requests a minute. Good answers are remembered, so repeated questions cost nothing. Uploads over 8 MB and files that are not images are refused.

## The image model

ResNet18, fine-tuned to tell four classes apart: Healthy, Coccidiosis, Salmonella and Newcastle disease. The weights are hosted on Hugging Face (`HilaryBuilds/poultry-disease-classifier`) and are downloaded when the server starts if they are not already present. Newcastle disease is the least reliable class in testing, so results for it carry an extra warning.

## Run it locally

```bash
python -m venv venv
source venv/Scripts/activate      # on Windows with Git Bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000/docs to try the endpoints.

To test the chatbot on its own against the real language model, run `python test_chatbot.py`.

## Settings

| Variable | Needed | What it does |
|---|---|---|
| `GROQ_API_KEY` | For AI explanations | Your free Groq key. Without it, explanations come straight from the knowledge base |
| `GROQ_MODEL` | No | Defaults to `openai/gpt-oss-20b` |
| `GROQ_MAX_PER_MIN` | No | The shared language model budget per minute. Defaults to 24 |

## Deploy on Render

Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`. Add `GROQ_API_KEY` under Environment. If a deploy times out with "no open ports detected", run Manual Deploy again, and consider adding `--extra-index-url https://download.pytorch.org/whl/cpu` as the first line of `requirements.txt` for a lighter install.

## Files

| File | What it holds |
|---|---|
| `main.py` | The API and the request limits |
| `chatbot_utils.py` | Explanations, follow-up answers and the safety checks |
| `model_utils.py` | Loads the image model |
| `knowledge_base.json` | The verified facts and follow-up questions for each result |
| `test_chatbot.py` | A live test of the chatbot |

## Limits

This is a guide from a photo, not a vet's diagnosis. Results depend on photo quality, and the free services it runs on can be slow to wake up and have usage limits. Editing `knowledge_base.json` changes what the bot can say, so changes should be checked against a reliable source.

## Credits

Built by Hilary Orefo as a personal learning project. Knowledge base sources: Ohio Department of Health, King County Public Health, dvm360, UMN Extension, the Merck Veterinary Manual and peer-reviewed zoonotic disease research.
