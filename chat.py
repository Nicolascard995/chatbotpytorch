"""
This script loads a pre-trained PyTorch model and uses it to chat with the user. It uses the intents.json file to understand the user's input and generate a response. 

The script imports the following modules:
    - random
    - json
    - torch
    - NeuralNet from model.py
    - bag_of_words and tokenize from nltk_utils.py

The script defines the following variables:
    - device: the device used to run the model (either 'cuda' or 'cpu')
    - intents: a dictionary containing the intents and their corresponding tags and responses
    - FILE: the path to the pre-trained model file
    - data: a dictionary containing the pre-trained model's input size, hidden size, output size, all words, tags, and model state
    - input_size: the input size of the pre-trained model
    - hidden_size: the hidden size of the pre-trained model
    - output_size: the output size of the pre-trained model
    - all_words: a list of all the words in the intents.json file
    - tags: a list of all the tags in the intents.json file
    - model_state: the state of the pre-trained model
    - model: an instance of the NeuralNet class with the pre-trained model's input size, hidden size, and output size
    - bot_name: the name of the chatbot

The script then enters a while loop that prompts the user for input and generates a response based on the pre-trained model's predictions. If the model's prediction probability is greater than 0.75, the script randomly selects a response from the corresponding intent in the intents.json file. If the prediction probability is less than or equal to 0.75, the script responds with "I do not understand...".

To exit the while loop, the user can type "quit".
"""
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"
print("Let's chat! type 'quit' to exit")
while True:
    try:
        sentence = input('You: ')
        if sentence == "quit":
            break

        sentence = tokenize(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X).to(device)

        output = model(X)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]
        if prob.item() > 0.75:
            for intent in intents["intents"]:
                if tag == intent["tag"]:
                    print(f"{bot_name}: {random.choice(intent['responses'])}")
        else:
            print(f"{bot_name}: I do not understand...")
    except:
        print(f"{bot_name}: An error occurred. Please try again.")