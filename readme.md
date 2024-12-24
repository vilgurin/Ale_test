## Setup & Installation

1. Create and activate Conda environment:
```bash
conda create -n "env name"
conda activate "env name"
```

2. Install PyTorch with CUDA support (adjust based on your CUDA version):
```bash
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install other dependencies:
```bash
pip install transformers soundfile jiwer scipy
```

This usually works too:
```bash
pip install torch transformers soundfile jiwer scipy
```
If you want to use requirements.txt:
```bash
pip install -r requirements.txt
```


4. Prepare your files:
- Input audio: `input_audio.wav`
- Reference text: `ground_truth.txt`

## Usage

Run evaluation:
```bash
python transcribe.py
```

Results are saved to `metrics.log` in JSON format.

## Interpreting Results

### Word Error Rate (WER)
- WER = (Substitutions + Deletions + Insertions) / Total Words
- Lower is better
- Typical ranges:
  - < 5%: Excellent
  - 5-10%: Good
  - 10-20%: Moderate
  - > 20%: Poor

Example:
```json
{
    "timestamp": "2024-12-24 09:30:45",
    "model": "openai/whisper-large-v3-turbo",
    "latency": 1.234,  // seconds
    "wer": 0.123      // 12.3% error rate
}
```

### Latency
- Time of inference of one sample
- Measured in seconds
- Reported as average over n runs
