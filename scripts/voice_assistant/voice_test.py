#!/usr/bin/env python3
"""
Voice Assistant Test - Talk and Listen

A simple conversational test for the voice assistant.
Uses NVIDIA Riva for ASR/TTS and Llama for conversation.

Usage:
    python voice_test.py
    python voice_test.py --no-llm  # Use simple responses without LLM
"""

import argparse
import io
import queue
import threading
import time
import wave

import numpy as np
import requests


class RivaTTS:
    """NVIDIA Riva Text-to-Speech."""
    
    def __init__(self, server: str = "localhost:50051"):
        self.server = server
        
        try:
            import riva.client
            
            self.auth = riva.client.Auth(uri=server)
            self.tts_service = riva.client.SpeechSynthesisService(self.auth)
            self.sample_rate = 22050
            
            print(f"[TTS] Connected to Riva TTS at {server}")
            
        except ImportError:
            print("[TTS] ERROR: nvidia-riva-client not installed")
            raise
        except Exception as e:
            print(f"[TTS] ERROR: Could not connect: {e}")
            raise
    
    def speak(self, text: str):
        """Synthesize and play speech."""
        import riva.client
        import sounddevice as sd
        
        if not text.strip():
            return
        
        print(f"[TTS] Speaking: {text}")
        
        try:
            responses = self.tts_service.synthesize_online(
                text,
                voice_name="English-US.Male-1",
                language_code="en-US",
                sample_rate_hz=self.sample_rate,
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
            )
            
            audio_chunks = []
            for response in responses:
                audio_chunks.append(np.frombuffer(response.audio, dtype=np.int16))
            
            if audio_chunks:
                audio = np.concatenate(audio_chunks)
                sd.play(audio, self.sample_rate)
                sd.wait()
                
        except Exception as e:
            print(f"[TTS] Error: {e}")


class RivaASR:
    """NVIDIA Riva streaming ASR."""
    
    def __init__(self, server: str = "localhost:50051", sample_rate: int = 16000):
        self.server = server
        self.sample_rate = sample_rate
        
        try:
            import riva.client
            
            self.auth = riva.client.Auth(uri=server)
            self.asr_service = riva.client.ASRService(self.auth)
            
            print(f"[ASR] Connected to Riva ASR at {server}")
            
        except ImportError:
            print("[ASR] ERROR: nvidia-riva-client not installed")
            raise
    
    def transcribe_stream(self, audio_queue: queue.Queue, result_callback, stop_event: threading.Event):
        """Stream audio to Riva and get transcriptions."""
        import riva.client
        
        config = riva.client.StreamingRecognitionConfig(
            config=riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=self.sample_rate,
                language_code="en-US",
                max_alternatives=1,
                enable_automatic_punctuation=True,
                verbatim_transcripts=False,
            ),
            interim_results=True,
        )
        
        def audio_generator():
            while not stop_event.is_set():
                try:
                    chunk = audio_queue.get(timeout=0.1)
                    if chunk is None:
                        break
                    yield chunk
                except queue.Empty:
                    continue
        
        try:
            responses = self.asr_service.streaming_response_generator(
                audio_chunks=audio_generator(),
                streaming_config=config,
            )
            
            for response in responses:
                if stop_event.is_set():
                    break
                if not response.results:
                    continue
                
                for result in response.results:
                    if result.alternatives:
                        transcript = result.alternatives[0].transcript
                        is_final = result.is_final
                        result_callback(transcript, is_final)
                        
        except Exception as e:
            if not stop_event.is_set():
                print(f"[ASR] Stream error: {e}")


class AudioCapture:
    """Capture audio from microphone."""
    
    def __init__(self, sample_rate: int = 16000, chunk_duration: float = 0.1):
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * chunk_duration)
        self.audio_queue = queue.Queue()
        self.running = False
        self.stream = None
    
    def start(self):
        """Start audio capture."""
        import sounddevice as sd
        
        self.running = True
        
        def callback(indata, frames, time_info, status):
            if status:
                print(f"[Audio] {status}")
            if self.running:
                self.audio_queue.put(bytes(indata))
        
        self.stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_size,
            dtype="int16",
            channels=1,
            callback=callback,
        )
        self.stream.start()
        print(f"[Audio] Microphone started")
    
    def stop(self):
        """Stop audio capture."""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.audio_queue.put(None)
    
    def pause(self):
        """Pause audio capture (while speaking)."""
        self.running = False
    
    def resume(self):
        """Resume audio capture."""
        self.running = True
    
    def get_queue(self) -> queue.Queue:
        return self.audio_queue


class ConversationAI:
    """Conversational AI using Llama via NIM."""
    
    SYSTEM_PROMPT = """You are JAX, the AI assistant for Spark Pack, an automated fulfillment company.
You manage multiple OpenArm robots used for automated packing operations.

Your role is to help operators train and control the robots through voice commands.
Keep responses short, professional, and conversational (1-2 sentences max).

Available commands you can help with:
- Start/stop recording demonstrations for robot training
- Spawn objects (cubes, mugs, fruits) for packing practice
- Reset/clear objects from the workspace
- Set task prompts describing what the robot should learn

About Spark Pack:
- Automated fulfillment company
- Uses OpenArm bimanual robots for packing
- Training robots through teleoperation demonstrations

Example interactions:
User: "Hello"
JAX: "Hello! JAX online. Ready to assist with your packing demonstration."

User: "What can you do?"
JAX: "I manage the OpenArm robots here at Spark Pack. I can help you record training demos, spawn items for packing practice, and set task descriptions."

User: "How do I start recording?"
JAX: "Just say 'start recording' and I'll capture your demonstration for robot training."

User: "Who are you?"
JAX: "I'm JAX, the AI assistant for Spark Pack's robotic fulfillment system."
"""

    def __init__(self, device: str = "cuda:0", use_llm: bool = True, nim_url: str = "http://localhost:8000"):
        self.device = device
        self.use_llm = use_llm
        self.nim_url = nim_url
        self.nim_available = False
        self.conversation_history = []
        
        if use_llm:
            self._check_nim()
    
    def _check_nim(self):
        """Check if NIM server is available."""
        print("[AI] Connecting to NIM server...")
        self.nim_model = "meta/llama-3.1-8b-instruct"  # default
        
        try:
            response = requests.get(f"{self.nim_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                model_id = models.get("data", [{}])[0].get("id", "unknown")
                self.nim_model = model_id
                print(f"[AI] NIM connected: {model_id}")
                self.nim_available = True
            else:
                raise Exception(f"NIM returned status {response.status_code}")
                
        except Exception as e:
            print(f"[AI] WARNING: NIM not available: {e}")
            print("[AI] Using simple responses")
            print("[AI] Start NIM with: ./scripts/voice_assistant/nim_start.sh")
            self.use_llm = False
    
    def respond(self, user_text: str) -> str:
        """Generate a response to user input."""
        if not user_text.strip():
            return ""
        
        if self.use_llm and self.nim_available:
            return self._respond_with_llm(user_text)
        else:
            return self._respond_simple(user_text)
    
    def _respond_with_llm(self, user_text: str) -> str:
        """Generate response using Llama via NIM."""
        self.conversation_history.append({"role": "user", "content": user_text})
        
        # Keep history short
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        messages = [{"role": "system", "content": self.SYSTEM_PROMPT}] + self.conversation_history
        
        try:
            response = requests.post(
                f"{self.nim_url}/v1/chat/completions",
                json={
                    "model": self.nim_model,
                    "messages": messages,
                    "max_tokens": 100,
                    "temperature": 0.7,
                },
                timeout=10,
            )
            
            if response.status_code != 200:
                return self._respond_simple(user_text)
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            self.conversation_history.append({"role": "assistant", "content": content})
            
            return content
            
        except Exception as e:
            print(f"[AI] NIM error: {e}")
            return self._respond_simple(user_text)
    
    def _respond_simple(self, user_text: str) -> str:
        """Simple keyword-based responses."""
        text = user_text.lower()
        
        if any(w in text for w in ["hello", "hi", "hey"]):
            return "Hello! JAX here, ready to assist with Spark Pack operations."
        
        if any(w in text for w in ["how are you", "what's up"]):
            return "All systems operational. Ready when you are."
        
        if "who are you" in text or "your name" in text:
            return "I'm JAX, the AI assistant for Spark Pack's robotic fulfillment system."
        
        if "what can you do" in text or "help" in text:
            return "I manage the OpenArm robots. I can record training demos, spawn items, and set task descriptions."
        
        if "start record" in text:
            return "Recording started. Capturing demonstration for robot training."
        
        if "stop record" in text or "end record" in text:
            return "Recording complete. Data saved for training."
        
        if "spawn" in text or "drop" in text:
            return "Spawning item for packing practice."
        
        if "reset" in text or "clear" in text:
            return "Clearing workspace."
        
        if "task" in text or "prompt" in text:
            return "Task description updated."
        
        if "thank" in text:
            return "Happy to help."
        
        if "bye" in text or "goodbye" in text:
            return "JAX standing by. Good luck with training."
        
        return "Standing by. How can I assist?"


class VoiceAssistantTest:
    """Interactive voice assistant test."""
    
    def __init__(self, riva_server: str = "localhost:50051", llm_device: str = "cuda:0", 
                 use_llm: bool = True, nim_url: str = "http://localhost:8000"):
        print("\n" + "=" * 50)
        print("VOICE ASSISTANT TEST")
        print("=" * 50)
        
        print("\n[1/4] Initializing TTS...")
        self.tts = RivaTTS(server=riva_server)
        
        print("\n[2/4] Initializing ASR...")
        self.asr = RivaASR(server=riva_server)
        
        print("\n[3/4] Initializing AI (NIM)...")
        self.ai = ConversationAI(device=llm_device, use_llm=use_llm, nim_url=nim_url)
        
        print("\n[4/4] Initializing audio capture...")
        self.audio = AudioCapture()
        
        self.running = False
        self.processing = False
        self.current_transcript = ""
        self.last_speech_time = 0
        self.silence_threshold = 1.5  # seconds
        self.stop_event = threading.Event()
        
        print("\n" + "=" * 50)
        print("READY - Start talking!")
        print("=" * 50)
        print("\nTry saying:")
        print("  - 'Hello'")
        print("  - 'What can you do?'")
        print("  - 'How do I start recording?'")
        print("\nPress Ctrl+C to quit\n")
    
    def _on_transcript(self, text: str, is_final: bool):
        """Handle transcription results."""
        if self.processing:
            return
        
        self.last_speech_time = time.time()
        
        if is_final and text.strip():
            self.current_transcript = text
            self._process_utterance(text)
        else:
            if text.strip():
                print(f"\r[You] {text}...          ", end="", flush=True)
    
    def _process_utterance(self, text: str):
        """Process a complete utterance and respond."""
        self.processing = True
        
        print(f"\r[You] {text}                    ")
        
        # Pause listening while we respond
        self.audio.pause()
        
        # Generate response
        response = self.ai.respond(text)
        
        if response:
            print(f"[Bot] {response}")
            self.tts.speak(response)
        
        # Resume listening
        self.audio.resume()
        self.processing = False
    
    def run(self):
        """Run the voice assistant test."""
        self.running = True
        self.stop_event.clear()
        
        # Greet the user
        self.tts.speak("JAX online. Ready to assist with Spark Pack operations.")
        
        # Start audio capture
        self.audio.start()
        
        # Start ASR in separate thread
        asr_thread = threading.Thread(
            target=self.asr.transcribe_stream,
            args=(self.audio.get_queue(), self._on_transcript, self.stop_event),
            daemon=True,
        )
        asr_thread.start()
        
        try:
            while self.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n[INFO] Shutting down...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the assistant."""
        self.running = False
        self.stop_event.set()
        self.audio.stop()
        self.tts.speak("Goodbye!")
        print("[INFO] Voice assistant stopped")


def main():
    parser = argparse.ArgumentParser(description="Voice Assistant Test")
    parser.add_argument("--riva-server", type=str, default="localhost:50051",
                        help="Riva gRPC server address")
    parser.add_argument("--llm-device", type=str, default="cuda:0",
                        help="Device for Llama model (unused with NIM)")
    parser.add_argument("--nim-url", type=str, default="http://localhost:8000",
                        help="NIM server URL for Llama inference")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM, use simple responses")
    args = parser.parse_args()
    
    assistant = VoiceAssistantTest(
        riva_server=args.riva_server,
        llm_device=args.llm_device,
        nim_url=args.nim_url,
        use_llm=not args.no_llm,
    )
    assistant.run()


if __name__ == "__main__":
    main()
