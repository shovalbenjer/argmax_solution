import nltk

# Download required NLTK data for ingredient-parser normalization
for resource in ["punkt", "wordnet"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
        print(f"NLTK resource '{resource}' already present.")
    except LookupError:
        print(f"Downloading NLTK resource: {resource}")
        nltk.download(resource) 