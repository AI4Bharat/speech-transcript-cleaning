# Speech Transcript Cleaning

Perform cleaning and normalization to standardize speech transcripts (train and test) across datasets.

We perform the following steps -
1. Remove puctuations
2. Normalize the characters

Note: We do not do the following steps for speech transcripts but might be useful for LMs used for speech models.
1. Remove sentences having Out of Dictionary characters.
2. Convert num to word

We also provide language-specific character [dictionaries](dicts) to remove OOD characters for each language. We recommend using the same dictionaries for training ASR models.

### Installation
```
pip install pandas
pip install indic-nlp-library
```

### Usage

`python cleaning.py <input_file> <output_file> <lang_code>`

```
input_file: file containing sentences (one sentence per line)
output_file: path to output file
lang_code: language code in ISO format
```

### License

Released under the MIT license


