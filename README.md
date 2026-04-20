# AIHC Chatbot

This project has two main uses:

- a local chatbot that takes a structured symptom summary and returns ranked diagnostic support
- an internal experiment runner for evaluating the same pipeline on the local datasets

## What You Need

- Python 3.11 or newer
- the local dataset files in `data/synthea/` and `data/ebm_nlp/`
- an OpenAI API key in `.env`

Minimum data files:

- `data/synthea/patients.csv`
- `data/synthea/conditions.csv`
- `data/ebm_nlp/knowledge_snippets.csv`

## Setup

From the project root:

```powershell
python -m pip install -r requirements.txt
```

Create or update [.env](</c:/Users/anjen/Desktop/project/anjenn/AIHC-chatbot/.env>) with:

```text
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5.4-mini
```

`CHATBOT_MODEL` is optional. If you set it, the web app will use that model instead of `OPENAI_MODEL`.

## Run The Chatbot

Start the local web app:

```powershell
python -m uvicorn chatbot_app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

The chatbot expects a structured symptom message. If the input is too short or too vague, it will show the intake template again.

## Run Internal Experiments

This runs the internal pipeline without the chatbot interface:

```powershell
python -m src.run_experiments
```

Results are written to `outputs/`.

## Generate Sample Cases For Chatbot Tests

This creates paste-ready test cases from the dataset, with the real diagnosis label saved separately:

```powershell
python -m src.sample_generator --split test --per-label 2 --output-prefix chatbot_experiment_samples_test
```

Files are written to `outputs/`.

## Main Files

- [chatbot_app/main.py](</c:/Users/anjen/Desktop/project/anjenn/AIHC-chatbot/chatbot_app/main.py>) starts the local web app.
- [src/chatbot_pipeline.py](</c:/Users/anjen/Desktop/project/anjenn/AIHC-chatbot/src/chatbot_pipeline.py>) runs the structured intake and reasoning pipeline.
- [src/run_experiments.py](</c:/Users/anjen/Desktop/project/anjenn/AIHC-chatbot/src/run_experiments.py>) runs the internal batch experiments.
- [src/sample_generator.py](</c:/Users/anjen/Desktop/project/anjenn/AIHC-chatbot/src/sample_generator.py>) creates sample chatbot inputs from the dataset.
- `dev/` holds legacy drafts and old development reports.

## Notes

- The ranked outputs are support signals, not confirmed medical diagnoses.
- `outputs/` is for generated experiment files.
- `dev/archive/` keeps older code and drafts that are no longer part of the active app.
