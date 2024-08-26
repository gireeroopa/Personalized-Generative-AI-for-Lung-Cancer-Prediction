import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import speech_recognition as sr
from difflib import SequenceMatcher
import pyttsx3
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import os
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
import IPython.display

# Initialize Tkinter GUI window
window = tk.Tk()
window.title("Integrated Input Processing")

# Initialize speech recognizer


# Load icons
mic_icon = ImageTk.PhotoImage(Image.open(r"C:\Users\nimes\Downloads\audio.jpeg").resize((30, 30)))
video_icon = ImageTk.PhotoImage(Image.open(r"C:\Users\nimes\Downloads\video.jpeg").resize((30, 30)))
image_icon = ImageTk.PhotoImage(Image.open(r"C:\Users\nimes\Downloads\image.jpeg").resize((30, 30)))
text_icon = ImageTk.PhotoImage(Image.open(r"C:\Users\nimes\Downloads\text.jpeg").resize((30, 30)))


# Function to initialize speech synthesis engine and speak text
# Function to initialize speech synthesis engine and speak text
def speak_text(text, rate=115):
    engine = pyttsx3.init()
    engine.setProperty('rate', rate)
    engine.say(text)
    engine.runAndWait()

# Preprocess conversations
conversations = [
    ("hi how are you", "Hello How may I help you?"),
    ("Can you tell me a joke", "Sure! Why don't scientists trust atoms? Because they make up everything!"),
    ("Goodbye!", "bye! Have a great day"),
    ("Who are you", "I am your virtual assistant."),
    ("What are the symptoms of asthma?",
     "Shortness of breath, wheezing, chest tightness, coughing, especially at night or early morning, are common symptoms of asthma"),
    ("What causes COPD?",
     "Chronic obstructive pulmonary disease (COPD) is caused by long-term exposure to irritants that damage the lungs. This can be from smoking, air pollution, or occupational dusts and chemicals"),
    ("How is lung cancer diagnosed?",
     "Chest X-rays, CT scans, and biopsies are some methods used to diagnose lung cancer. Other tests may also be used, like sputum cytology or PET scans."),
    ("OK", "Great Is the information useful?"),
    ("Yes", "Happy to help you"),
    ("What are the risk factors that cause lung cancer?",
     "Smoking is the leading risk factor for lung cancer. Exposure to secondhand smoke, radon gas, and certain air pollutants also increase risk. A family history of lung cancer can be a factor as well."),
    ("Can pneumonia be cured?",
     "Yes, most cases of pneumonia can be cured with antibiotics. Early diagnosis and treatment are key for a full recovery"),
    ("What are the types of lung disease?",
     "Asthma, COPD, Lung cancer, Pneumonia, and Pulmonary fibrosis are some common lung diseases. Others include cystic fibrosis, bronchiectasis, and pulmonary sarcoidosis"),
    ("Can I prevent lung disease?",
     "Avoiding smoking, reducing air pollution exposure, and getting regular checkups can help prevent lung disease. Getting vaccinated against pneumonia and influenza can also offer protection"),
    ("What are the signs of a lung infection?",
     "Fever, cough, shortness of breath, chest pain, and fatigue can be signs of a lung infection. If you experience these symptoms, consulting a doctor for a diagnosis and treatment is crucial"),
    ("Is lung cancer more common in men or women today",
     " Lung cancer has surpassed breast cancer as the leading cause of cancer death in women [1]. This highlights a major public health concern. The rise in lung cancer rates among women is likely due to increased smoking rates, particularly among younger women who began smoking at a similar age to men (15-18 years old). The latency period between initial tobacco exposure and lung cancer diagnosis can be 25-30 years, meaning women who started smoking young are now reaching the age where these cancers are developing. Additionally, with women delaying childbirth compared to previous generations, a lung cancer diagnosis can significantly impact young children in the family"),
    ("Besides smoking, what are other things that can increase your risk of lung cancer",
     "While smoking is the primary cause, other factors can contribute to lung cancer risk. Exposure to environmental pollutants and workplace toxins can play a role. Regulations are in place to minimize these risks by monitoring dust and harmful particles in the environment and by protecting non-smokers from secondhand smoke in public spaces "),
    (" Is there a genetic link to lung cancer",
     " Unlike some cancers, there isn't a direct inheritance pattern for lung cancer. Some families may have a higher incidence due to unknown genetic factors, but there's no evidence that lung cancer itself is directly passed down through generations"),
    ("What are some warning signs of lung cancer",
     "It's important to note that lung cancer itself can be a sign of a more advanced stage of the disease. However, some common symptoms include a persistent cough that doesn't respond to medication, difficulty breathing (wheezing), unexplained weight loss, and persistent pain, often in the chest but potentially in other areas as well. In some cases, lung cancer may cause fever or neurological changes"),
    ("If lung cancer has progressed to a later stage, which organs are most commonly affected",
     "The organs most likely to be impacted by advanced lung cancer depend on the patient's specific symptoms. These can include the lungs themselves, lymph nodes in the chest cavity (mediastinum), bones, liver, brain, and adrenal glands located near the kidneys"),
    ("What is lung cancer",
     "Lung cancer is a disease in which certain cells in the lungs become abnormal and multiply uncontrollably to form a tumor. Lung cancer may not cause signs or symptoms in its early stages. Some people with lung cancer have chest pain, frequent coughing, blood in the mucus, breathing problems, trouble swallowing or speaking, loss of appetite and weight loss, fatigue, or swelling in the face or neck"),
    ("Is lung cancer genetic or not",
     "No, lung cancer is not directly genetic You cannot inherit lung cancer itself from a parent. There isn't a simple yes or no answer because genetics might play a small role by making some people more susceptible but it's not the main cause. Smoking is the overwhelming risk factor for lung cancer"),
    ("people of age around 11-20 years to have lung cancer ",
     "Less common, but possible People with a history of heavy smoking for many years are at a much higher risk within this age range, even if they quit smoking earlier in life."),
    ("people of age around 1 to 10 years to have lung cancer",
     "It is very rare but possible to get due to genetics"),
    ("people of age around 55 to 100 years to have lung cancer", "there are High risks"),
    ("person does not smoke but has lung cancer", "it can be genetic")
]

processed_conversations = [
    (re.sub(r"[^\w\s]", "", pair[0].lower()), re.sub(r"[^\w\s]", "", pair[1].lower())) for pair in
    conversations]

user_queries, model_responses = zip(*processed_conversations)

# Function to generate response
def generate_response(text):
    max_similarity = -1
    best_match_index = -1

    # Find closest match
    for i, query in enumerate(user_queries):
        similarity = SequenceMatcher(None, query, text).ratio()
        if similarity > max_similarity:
            max_similarity = similarity
            best_match_index = i

    # Generate response
    if best_match_index != -1:
        response_text = model_responses[best_match_index]
    else:
        response_text = "Sorry, I didn't understand. Can you repeat that?"

    return response_text

# Function to handle voice input
def handle_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as src:
        audio_data = recognizer.listen(src)
        try:
            text = recognizer.recognize_google(audio_data).lower()
            print("You said:", text)
            response = generate_response(text)
            print("Response:", response)
            speak_text(response)
        except sr.UnknownValueError:
            print("Sorry, could not understand audio.")
        except sr.RequestError as e:
            print("Could not request results; {0}".format(e))
        except KeyboardInterrupt:
            print("Interrupted by user. Exiting...")
# Function to capture image and process using image processing model (GAN)
def capture_and_process_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Pass the image to the image processing model (GAN)
        # Your image processing model code goes here

        class Generator(nn.Module):
            def _init_(self, latent_dim, image_shape):
                super(Generator, self)._init_()
                self.latent_dim = latent_dim
                self.image_shape = image_shape

                self.model = nn.Sequential(
                    nn.Linear(latent_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 2048),  # Increased complexity
                    nn.ReLU(),
                    nn.Linear(2048, np.prod(image_shape)),
                    nn.Tanh()
                )

            def forward(self, z):
                img = self.model(z)
                img = img.view(img.size(0), *self.image_shape)
                return img

        # Discriminator class definition
        class Discriminator(nn.Module):
            def _init_(self, image_shape, num_classes):
                super(Discriminator, self)._init_()
                self.image_shape = image_shape
                self.num_classes = num_classes

                self.model = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(np.prod(image_shape), 1024),
                    nn.LeakyReLU(0.2),
                    nn.Linear(1024, 512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, 256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, num_classes)  # Output should reflect the number of classes
                )

            def forward(self, img):
                validity = self.model(img)
                return validity

        # Define the transform for the dataset
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load Custom Dataset
        dataset = ImageFolder(root=r"C:\Users\nimes\Downloads\archive\training_data\The IQ-OTHNCCD lung cancer dataset",
                              transform=transform)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

        # Model hyperparameters
        latent_dim = 100
        image_shape = (3, 32, 32)
        learning_rate = 0.0002
        beta1 = 0.5
        beta2 = 0.999
        num_classes = len(dataset.classes)

        # Initialize the generator and discriminator
        generator = Generator(latent_dim, image_shape)
        discriminator = Discriminator(image_shape, num_classes)

        # Initialize weights
        def weights_init(m):
            classname = m._class.name_

            if classname.find('Linear') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                nn.init.constant_(m.bias.data, 0)

        generator.apply(weights_init)
        discriminator.apply(weights_init)

        # Define the optimizers
        optimizer_G = optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))
        optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

        # Define the loss function
        adversarial_loss = nn.CrossEntropyLoss()

        # Training parameters
        num_epochs = 1

        # Output directory for generated images
        os.makedirs("generated_images", exist_ok=True)

        # Training loop
        def save_image(param, param1, nrow, normalize):
            pass

        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):
                real_imgs, labels = data
                batch_size = real_imgs.size(0)

                # Adversarial ground truths
                real_labels = labels.type(torch.LongTensor)
                fake_labels = torch.randint(0, num_classes, (batch_size,), dtype=torch.long)

                # Train Generator
                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = torch.randn(batch_size, latent_dim)

                # Generate a batch of images
                gen_imgs = generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gen_imgs), fake_labels)

                g_loss.backward()
                optimizer_G.step()

                # Train Discriminator
                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                d_real_loss = adversarial_loss(discriminator(real_imgs), real_labels)
                d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake_labels)
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                print(
                    f"[Epoch {epoch + 1}/{num_epochs}] [Batch {i + 1}/{len(dataloader)}] "
                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]"
                )

            # Save generated images for each epoch
            save_image(gen_imgs.data[:25], f"generated_images/epoch_{epoch + 1}.png", nrow=5, normalize=True)
            # Save input image
            save_image(real_imgs.data, f"generated_images/input_image_epoch_{epoch + 1}.png", nrow=8, normalize=True)

        print("Training finished.")

        # Function to test the model on new images
        def test_model(image_path):
            # Load the image and preprocess
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
            image = transform(Image.open(image_path)).unsqueeze(0)

            # Set model to evaluation mode
            generator.eval()
            discriminator.eval()

            # Perform inference
            with torch.no_grad():
                output = discriminator(image)
                _, predicted = torch.max(output, 1)
                predicted_class = dataset.classes[predicted.item()]
                print("Predicted class:", predicted_class)

                # Reverse normalization and display input image with increased clarity
                image = image.squeeze(0)  # Remove batch dimension
                image = image * 0.5 + 0.5  # Reverse normalization
                image = transforms.functional.adjust_brightness(image, 1.5)  # Increase brightness for clarity
                image = transforms.functional.to_pil_image(image)  # Convert Tensor back to PIL Image
                image.show()

        # Example usage:
        test_model(r"C:\Users\nimes\Downloads\abc.jpg")

# Function to process text input using text processing model (GPT)
def process_text_input(input_text):
    # Pass the text to the text processing model (GPT)
    # Your text processing model code goes here

    # Define your GPT model architecture
    class GPTModel(nn.Module):
        def _init_(self, vocab_size, embedding_size, hidden_size, num_layers, num_heads, max_sequence_length):
            super(GPTModel, self)._init_()
            self.embedding = nn.Embedding(vocab_size, embedding_size)
            self.transformer_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(embedding_size, num_heads, hidden_size, dropout=0.1),
                num_layers
            )
            self.fc = nn.Linear(embedding_size, vocab_size)

        def forward(self, input_ids):
            # Add batch dimension if it's not present
            if len(input_ids.shape) == 1:
                input_ids = input_ids.unsqueeze(0)

            embeddings = self.embedding(input_ids)
            embeddings = embeddings.permute(1, 0, 2)
            output = self.transformer_encoder(embeddings)
            output = output.permute(1, 0, 2)
            output_logits = self.fc(output)
            return output_logits

    # Define a list of input-output pairs (replace with your actual data)
    input_output_pairs = [
        ("hii", "how are you"),
        ("hello", "how are you"),
        ("hey", "how are you"),
        # Add more word-level input-output pairs
    ]

    # Create word-to-index and index-to-word mappings
    word_to_index = {}  # Map words to unique integer IDs
    index_to_word = []  # Map integer IDs back to words
    for pair in input_output_pairs:
        for word in pair[0].lower().strip().split():
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
                index_to_word.append(word)
        for word in pair[1].lower().strip().split():
            if word not in word_to_index:
                word_to_index[word] = len(word_to_index)
                index_to_word.append(word)

    # Define functions to convert text to tensor and vice versa
    def text_to_tensor(text, word_to_index):
        tokens = text.lower().strip().split()
        tensor = torch.tensor([word_to_index[word] for word in tokens])
        return tensor.unsqueeze(0)  # Add batch dimension

    def tensor_to_text(tensor, index_to_word):
        words = [index_to_word[idx.item()] for idx in tensor.squeeze()]
        return ' '.join(words)

    # Define functions to pad or truncate sequences
    def pad_sequence(sequence, max_length, padding_token):
        if len(sequence) < max_length:
            sequence += [padding_token] * (max_length - len(sequence))
        elif len(sequence) > max_length:
            sequence = sequence[:max_length]
        return sequence

    # Convert input-output pairs to tensors with padding
    max_sequence_length = max(len(pair[0].split()) for pair in input_output_pairs)
    input_tensors = [
        text_to_tensor(' '.join(pad_sequence(pair[0].lower().strip().split(), max_sequence_length, '<PAD>')),
                       word_to_index) for pair in input_output_pairs]
    target_tensors = [
        text_to_tensor(' '.join(pad_sequence(pair[1].lower().strip().split(), max_sequence_length, '<PAD>')),
                       word_to_index) for pair in input_output_pairs]

    # Set hyperparameters
    vocab_size = len(word_to_index)
    embedding_size = 64
    hidden_size = 128
    num_layers = 2
    num_heads = 2
    learning_rate = 0.001
    num_epochs = 1000  # Train for a larger number of epochs

    # Instantiate the model
    model = GPTModel(vocab_size, embedding_size, hidden_size, num_layers, num_heads, max_sequence_length)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize a variable to keep track of the current epoch
    current_epoch = 0

    # Check if there is a saved model checkpoint
    checkpoint_path = "gpt_model_checkpoint.pth"
    if os.path.exists(checkpoint_path):
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        current_epoch = checkpoint['epoch'] + 1  # Start from the next epoch

    # Training loop
    for epoch in range(current_epoch, num_epochs):
        total_loss = 0
        for input_tensor, target_tensor in zip(input_tensors, target_tensors):
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            output_logits = model(input_tensor)

            # Compute the loss
            loss = criterion(output_logits.view(-1, vocab_size), target_tensor.view(-1))

            # Backward pass
            loss.backward()

            # Update the parameters
            optimizer.step()

            total_loss += loss.item()

        # Print loss for every epoch
        formatted_loss = '{:.8f}'.format(total_loss / len(input_tensors))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {formatted_loss}')

        # Save the model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': total_loss / len(input_tensors)
        }, checkpoint_path)

        # Check if the total loss is very small (close to 0)
        if total_loss / len(input_tensors) <= 1e-8:
            print("Loss successfully reached a very small value. Training terminated.")
            break

    # Generate relevant response based on user input
    def generate_response(user_input, model, word_to_index, index_to_word, max_length=100):
        input_text = user_input.lower().strip()
        tokens = re.findall(r'\b\w+\b', input_text)

        if any(token in input_text for token in ['hi', 'hello', 'hey']):
            return "Hello! How may I assist you?"

        # Check if all tokens are in the vocabulary
        unknown_tokens = [token for token in tokens if token not in word_to_index]
        if unknown_tokens:
            print(f"I'm sorry, the following tokens are not in the vocabulary: {', '.join(unknown_tokens)}")
            return "Please provide a different input."

        # Convert tokens to tensor indices
        input_tensor = text_to_tensor(' '.join(pad_sequence(tokens, max_length, '<PAD>')), word_to_index)

        # Forward pass through the model
        output_logits = model(input_tensor)

        # Decode the output tensor into text
        output_text = tensor_to_text(torch.argmax(output_logits, dim=-1), index_to_word)

        return output_text

    # Test the model with user input
    pass

#function to get video input
def input_video():
    def extract_video_id(url):
        match = re.search(r"watch\?v=(\S+)", url)
        if match:
            return match.group(1)
        else:
            return None

    # Function to fetch and summarize the transcript of a YouTube video
    def fetch_and_summarize(youtube_video):
        # Extract video ID from the URL
        video_id = extract_video_id(youtube_video)
        if not video_id:
            print("Invalid YouTube URL. Please provide a valid YouTube video URL.")
            return

        # Display YouTube video
        IPython.display.display(IPython.display.YouTubeVideo(video_id))

        # Fetch transcript
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        result = " ".join([i['text'] for i in transcript])

        # Initialize summarizer
        summarizer = pipeline("summarization")

        # Summarize transcript
        summarized_text = []
        num_iters = int(len(result) / 1000) + 1
        for i in range(num_iters):
            start = i * 1000
            end = min((i + 1) * 1000, len(result))
            chunk = result[start:end]
            chunk_summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
            summarized_text.append(chunk_summary)
        summarized_text = " ".join(summarized_text)
        print("Summarized Text:", summarized_text)

    # Get YouTube video URL from user
    youtube_video = input("Enter the YouTube video URL: ")

    # Fetch and summarize the transcript
    fetch_and_summarize(youtube_video)
    pass

# Function to display input fetching GUI
def input_gui():
    input_window = tk.Toplevel(window)
    input_window.title("Input Fetching")

    # Function to handle text input
    def get_text_input():
        input_text = text_entry.get()
        response = generate_response(input_text)
        print("Bot:", response)
        input_window.destroy()

    # Function to handle audio input
    def get_audio_input():
        handle_voice_input()
        input_window.destroy()

    # Function to handle image input
    def get_image_input():
        capture_and_process_image()
        input_window.destroy()

    #function to handle video input
    def get_video_input():
        input_video()
        input_window.destroy()

    # Text input
    text_label = tk.Label(input_window, text="Enter text:")
    text_label.pack()
    text_entry = tk.Entry(input_window)
    text_entry.pack()
    text_button = tk.Button(input_window, image=text_icon, command=get_text_input)
    text_button.pack()

    # Audio input
    audio_button = tk.Button(input_window, image=mic_icon, command=get_audio_input)
    audio_button.pack()

    # Image input
    image_button = tk.Button(input_window, image=image_icon, command=get_image_input)
    image_button.pack()

    #video input
    video_button = tk.Button(input_window, image=video_icon, command=get_video_input)
    video_button.pack()

# Button to trigger input fetching GUI
input_button = tk.Button(window, text="Fetch Input", command=input_gui)
input_button.pack(pady=10)

window.mainloop()
