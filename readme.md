# Butler Assistant

An AI butler powered by LiveKit that provides voice interaction, visual assistance, and memory capabilities.

## Features
- Voice interaction using OpenAI GPT-4
- Visual assistance through efficient screen capture
- Memory of interactions and preferences
- Professional butler-like personality
- English communication with Thai phrases

## Prerequisites
- Python 3.9+
- OpenAI API key
- Deepgram API key
- LiveKit account and credentials

## Installation

1. Clone repository:
```bash
git clone [your-repo-url]
cd butler-project
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create `.env` file with your API keys:
```
OPENAI_API_KEY=your_key
DEEPGRAM_API_KEY=your_key
LIVEKIT_URL=your_livekit_url
LIVEKIT_API_KEY=your_livekit_key
LIVEKIT_API_SECRET=your_livekit_secret
```

## Usage
```bash
python butler.py dev
```

## Project Structure
```
butler-project/
├── .gitignore
├── requirements.txt
├── README.md
└── butler.py
```

## License
MIT License