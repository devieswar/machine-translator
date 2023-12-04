Machine Translation Test Model

This script provides a simple function, `test_model`, to perform machine translation using a pre-trained model. The function takes an input text in English and returns the translated text in Spanish.

`test_model` Function

The `test_model` function is designed to take an English input text, tokenize it, and use a pre-trained machine translation model to generate the corresponding Spanish translation.

Usage

1. **Load Pre-trained Model:**
   Ensure that you have a pre-trained machine translation model. The path to the model weights should be specified in the script.

2. **Run the Script:**
   Execute the script, and it will prompt you to enter an English sentence. After entering the sentence, the script will use the `test_model` function to generate the translation.

   ```bash
   python test_model.py
