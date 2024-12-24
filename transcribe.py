import time
import json
import torch
import soundfile as sf
from jiwer import wer
from datetime import datetime
from jiwer import transforms as tr
from scipy.signal import resample
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def log_metrics(log_file, file_path, latency_info, wer_value, model_id):
    """Logs metrics to a file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "model": model_id,
        "latency": latency_info,
        "wer": wer_value
    }
    
    with open(log_file, "a") as f:
        json.dump(log_entry, f)
        f.write("\n")



def transcribe_wav_file(file_path):
    """Transcribes the text from an input WAV file."""
    audio, sample_rate = sf.read(file_path)

    # Resample to 16KHz
    target_sample_rate = 16000
    if sample_rate != target_sample_rate:
        num_samples = int(len(audio) * target_sample_rate / sample_rate)
        audio = resample(audio, num_samples)

    result = pipe(audio)
    return result["text"]

def calculate_latency(file_path, num_runs=20):
    """Calculates average transcription latency over multiple runs."""
    audio, sample_rate = sf.read(file_path)

    # Resample to 16KHz
    target_sr = 16000
    if sample_rate != target_sr:
        num_samples = int(len(audio) * target_sr / sample_rate)
        audio = resample(audio, num_samples)
    # get audio duration
    len_audio = len(audio) / target_sr

    latencies = []
    for _ in range(num_runs):
        start_time = time.time()
        pipe(audio)
        end_time = time.time()
        latencies.append(end_time - start_time)

    average_latency = sum(latencies) / num_runs
    print(f"Latency: {average_latency:.2f} seconds for an audio with duration {len_audio}s")
    return average_latency


def preprocess_text(text):
    """Normalize text"""
    transform = tr.Compose([
        tr.ToLowerCase(),
        tr.ExpandCommonEnglishContractions(),
        tr.RemoveKaldiNonWords(),
        tr.RemoveWhiteSpace(replace_by_space=True),
        tr.RemoveMultipleSpaces(),
        tr.Strip(),
    ])
    
    return transform(text)

def calculate_wer(transcription, ground_truth_file):
    """Calculates the Word Error Rate between the transcription and ground truth."""
    with open(ground_truth_file, "r") as f:
        ground_truth = f.read().strip()
    

    transcription_norm = preprocess_text(transcription)
    ground_truth_norm = preprocess_text(ground_truth)
    
    error_rate = wer(ground_truth_norm, transcription_norm)
    print(f"Word Error Rate (WER): {error_rate:.2%}")
    
    return error_rate



if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    # Create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=30, 
        batch_size=1,     
        torch_dtype=torch_dtype,
        device=device,
    )

    wav_file_path = "input_audio.wav"  
    ground_truth_path = "ground_truth.txt"
    log_file = "metrics.log"

    try:
        transcription = transcribe_wav_file(wav_file_path)
        
        latency = calculate_latency(wav_file_path, num_runs=20)
        wer_rate = calculate_wer(transcription, ground_truth_path)
        
        log_metrics(log_file, wav_file_path, latency, wer_rate, model_id)

    except Exception as e:
        print("Error:", e)
