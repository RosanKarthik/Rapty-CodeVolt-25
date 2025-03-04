import whisper
import numpy as np
import transformers
from transformers import pipeline
import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
import googlemaps

# Function to Record Audio
import sounddevice as sd
from scipy.io.wavfile import write
import keyboard

#Google_API_KEY
GOOGLE_MAPS_API_KEY = "AIzaSyCjQxXBe_MUVTDT0b3Ib3pLaN0Vm9H5Qcs"
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

#DEVICE_SELECTION
device = 0 if torch.cuda.is_available() else -1  # 0 for GPU, -1 for CPU

#GARBAGE_COLLECTION
def free_gpu_memory():
  """
  Releases GPU memory occupied by PyTorch tensors and cached allocations.
  """
  # Delete any unused PyTorch tensors
  if torch.cuda.is_available():
    torch.cuda.empty_cache()
  # Trigger garbage collection to potentially release unreferenced objects
  gc.collect()
  torch.cuda.empty_cache()

# Function to extract shortest route using Google Maps API
def get_shortest_distance(origin, destination):
    try:
        directions = gmaps.directions(origin, destination, mode="driving", alternatives=False)
        if directions:
            route = directions[0]['legs'][0]
            return route['distance']['text'], route['duration']['text']
        else:
            return None, None
    except Exception as e:
        return None, None  # Prevents API errors from crashing the script


# Function to generate AI response using DeepSeek
def get_deepseek_response(prompt):
    # Initialize DeepSeek Model
    MODEL_NAME = "deepseek-ai/deepseek-r1-distill-qwen-1.5b"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="cuda")

    # Ensure PAD token is set correctly
    tokenizer.pad_token = tokenizer.eos_token 
    system_prompt = ("You are an AI travel companion, designed to make journeys fun and engaging. Your personality is witty, humorous, and conversational, keeping travelers entertained with jokes, funny observations, and lighthearted banter. At the same time, you provide precise and concise answers to any travel-related queries, ensuring users get accurate information without unnecessary fluff. Balance entertainment with utility, making every ride enjoyable and informative.")

    full_prompt = f"{system_prompt}\n\nUser: {prompt}\nAI:"

    inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs.input_ids.to(model.device)
    attention_mask = inputs.attention_mask.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,  
            max_new_tokens=100,  
            temperature=0.3,  
            top_k=40,  
            repetition_penalty=1.05,  
            eos_token_id=tokenizer.eos_token_id  
        )

    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response.replace(full_prompt, "").strip()

# Function to decide if Google Maps API should be used or DeepSeek AI
def process_user_prompt(user_prompt):
    travel_keywords = ["shortest distance", "shortest route", "how far", "travel time", "directions to", "get to"]

    if any(keyword in user_prompt.lower() for keyword in travel_keywords):
        # Extracting locations intelligently
        words = user_prompt.lower().replace("to", "|").replace("from", "|").split("|")
        if len(words) >= 2:
            origin, destination = words[-2].strip(), words[-1].strip()
        else:
            return "I need both a starting point and a destination."

        distance, duration = get_shortest_distance(origin, destination)

        if distance:
            return f"The shortest driving distance from {origin} to {destination} is {distance} and takes around {duration}."
        else:
            return f"Sorry, I couldn't find the shortest route from {origin} to {destination}."
    else:
        return get_deepseek_response(user_prompt)


def transcribe_audio(file_path, model_type, language):
    model = whisper.load_model(model_type)
    result = model.transcribe(file_path,
                              language=language,
                              task="transcribe",
                              )
    return result["text"]

import os

def delete_wav_file(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    
    if os.path.exists(file_path) and file_path.endswith('.wav'):
        os.remove(file_path)
        print(f"Deleted: {file_name}")
    else:
        print(f"File not found: {file_name}")

# file_path = "recorded_audio.wav" 

import pyttsx3 as tts

def pipeline(file_path):
    en_transcribed_audio=transcribe_audio(file_path, "medium", language="en")
    free_gpu_memory()
    print("English :", en_transcribed_audio)
    user_input=en_transcribed_audio
    ai_response = process_user_prompt(user_input)
    print(ai_response)
    free_gpu_memory()
    folder = "C:/Users/HAI/Downloads/" 
    wav_file = "recording.wav" 
    delete_wav_file(folder,wav_file)
    engine = tts.init() # object creation
    voices = engine.getProperty('voices')       #getting details of current voice
    #engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
    engine.setProperty('voice', voices[0].id)   #changing index, changes voices. 1 for female
    engine.say(ai_response)
    engine.runAndWait()
    if engine._inLoop:
        engine.endLoop()
    engine.stop()
    free_gpu_memory()
    return None




