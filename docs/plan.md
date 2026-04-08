# Plan

Historical notes and early implementation ideas. This is not the current API or
project plan.

## Steps:

> (Storing and reading utilities can mostly be done by LLMs and then checked.)

1. Load the data from William JSONL.
2. Decide which questions to run it on -> activations for each question, for each token in the response -> average across them.
3. Store persona vectors on disk (h5py), .safetensor (or simple .pt) -> upload to Hugging Face.
4. Utility to read from those (loading them from Hugging Face).
   4.5. Test which layers are more relevant.
5. Add additional relevant plots
6. Steering with those persona vectors, for example (decide layer; most likely mid layers).

---

### Simple extraction

- (system prompt)
- Question
  Response (....) -> persona_vector for each layer (averaged across tokens in the response)
  -> System prompt and question parts are ignored for this average

---

### PersonaGuess extraction

- (system prompt)

- Question 1:
- Response 1:

- Question 2:
- Response 2:

...

[Response1, Response2, ...] -> flatten tensor and then average
Average activation across all tokens in each of the responses

-> A similar approach can be taken also for Synth Persona.

> [!IMPORTANT]
> In general we want something flexibly which can **average across the tokens we specify** (eg. only the reponses or things like that)
> Something that is easy to use where I can pass a mask of those tokens perhaps

### TODO:

1. [ ] Extend code in functions to reuse what is presented inside the notebook.
2. [ ] Allow command-line arguments to pass different JSON files and read different formats from which we can have the system prompts, then run the rest.
3. [ ] Extend to more questions instead of the default one.
4. [ ] Extend to use the responses from Synth Persona and PersonaGuess without the need to regenerate the responses.
5. [ ] Update the main script to use argparse with flags like `--extract` and `--plot` to create plots or save the extracted persona vector to disk in a specified directory.
