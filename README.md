\# Wake Word Detector (Sliding Window)



Simple wake word detection system using the Google Speech Commands dataset.

The system detects a predefined wake word (e.g., "Omar") from continuous audio

using a sliding 1-second window approach.



\## Features

\- MFCC feature extraction

\- CNN-based binary classifier

\- Sliding window inference (1s window, 250ms hop)



\## Dataset

Google Speech Commands Dataset (v0.02)



\## How to Run



\### Install dependencies

```bash

pip install -r requirements.txt



