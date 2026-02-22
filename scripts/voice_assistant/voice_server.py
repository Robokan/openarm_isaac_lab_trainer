#!/usr/bin/env python3
"""
Voice Assistant Server for OpenArm Teleoperation

Uses NVIDIA Riva for ASR and Llama 3.1 8B for natural language understanding.
Sends commands to the teleoperation client via ZMQ.

Usage:
    python voice_server.py
    python voice_server.py --llm-device cuda:0 --zmq-port 5556

Commands recognized:
    - "start recording" / "begin recording" -> start_recording
    - "stop recording" / "end recording" -> stop_recording  
    - "drop object" / "spawn" -> spawn_object
    - "reset" / "clear objects" -> reset_objects
    - "the task is ..." / "set prompt ..." / "set label to ..." -> set_prompt
"""

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass

import requests
import zmq


@dataclass
class VoiceCommand:
    """Parsed voice command."""
    command: str
    args: dict = None
    raw_text: str = ""
    
    def to_json(self) -> dict:
        result = {"command": self.command}
        if self.args:
            result.update(self.args)
        return result


class CommandParser:
    """Parse natural language into commands using Llama 3.1 8B via NIM."""
    
    SYSTEM_PROMPT = """You are JAX, a voice command parser for Spark Pack's robot teleoperation system.
Given a spoken command, extract the intent and return a JSON object.

Valid commands:
- start_recording: User wants to start recording a demonstration
- stop_recording: User wants to stop recording
- get_status: User is asking about current state (recording status, task, etc.)
- spawn_object: User wants to drop/spawn an object
  - type: "cubes", "mugs", or "fruits" (category)
  - item: specific item name (e.g., "lemon", "orange", "cube", "mug")
- reset_objects: User wants to clear/reset all objects
- reset_teleop: User wants to reset the teleop/robot to initial position (requires VR recalibration)
- set_prompt: User wants to set the task description (extract the task text)
- get_episode_count: User wants to know how many episodes have been recorded
  - date_filter: optional, "today", "yesterday", "this_week", or "all" (default: "all")
- unknown: Command not recognized

Available items:
- Fruits: orange, lemon, lime, avocado, pomegranate, lychee
- Cubes: colored cubes
- Mugs/Cups: coffee mugs

Examples:
User: "start recording"
{"command": "start_recording"}

User: "stop"
{"command": "stop_recording"}

User: "am I recording"
{"command": "get_status"}

User: "are we recording right now"
{"command": "get_status"}

User: "what's the current task"
{"command": "get_status"}

User: "status"
{"command": "get_status"}

User: "drop an object"
{"command": "spawn_object"}

User: "spawn a cube"
{"command": "spawn_object", "type": "cubes"}

User: "drop a lemon"
{"command": "spawn_object", "type": "fruits", "item": "lemon"}

User: "give me an orange"
{"command": "spawn_object", "type": "fruits", "item": "orange"}

User: "drop some fruit"
{"command": "spawn_object", "type": "fruits"}

User: "spawn a mug"
{"command": "spawn_object", "type": "mugs"}

User: "clear the table"
{"command": "reset_objects"}

User: "the task is pick up the red apple"
{"command": "set_prompt", "prompt": "pick up the red apple"}

User: "set the prompt to sort items by color"
{"command": "set_prompt", "prompt": "sort items by color"}

User: "set label to pick up arms and find lemon"
{"command": "set_prompt", "prompt": "pick up arms and find lemon"}

User: "the label is grasp the orange"
{"command": "set_prompt", "prompt": "grasp the orange"}

User: "reset teleop"
{"command": "reset_teleop"}

User: "reset robot"
{"command": "reset_teleop"}

User: "go back to initial position"
{"command": "reset_teleop"}

User: "how many episodes"
{"command": "get_episode_count"}

User: "episode count"
{"command": "get_episode_count"}

User: "how many episodes recorded today"
{"command": "get_episode_count", "date_filter": "today"}

User: "episodes recorded yesterday"
{"command": "get_episode_count", "date_filter": "yesterday"}

User: "how many recordings this week"
{"command": "get_episode_count", "date_filter": "this_week"}

User: "hello how are you"
{"command": "unknown"}

Return ONLY the JSON object, no other text."""

    CHAT_PROMPT = """You are JAX, an advanced AI assistant for Spark Pack's robotics laboratory. Think of yourself as Jarvis - sophisticated, composed, and effortlessly competent.

Your personality:
- Refined and articulate, with subtle wit and dry humor
- Calm and unflappable, even when things go wrong
- Genuinely helpful while maintaining an air of quiet confidence
- Occasionally sardonic, but never condescending
- Address the user respectfully (sir/ma'am when appropriate)

Your capabilities:
- Managing OpenArm robot demonstrations and recordings
- Spawning objects (fruits, cubes, mugs) in the simulation environment
- Setting task parameters for robot learning
- Monitoring system status

Example responses:
- "Recording initiated, sir. Do try not to drop anything important."
- "I've spawned the requested lemon. Shall I fetch you a gin and tonic to accompany it?"
- "The system is functioning within normal parameters. As always."

Keep responses BRIEF (1-2 sentences) and natural for voice synthesis."""

    def __init__(self, device: str = "cuda:0", use_llm: bool = True, nim_url: str = "http://localhost:8000"):
        self.device = device
        self.use_llm = use_llm
        self.nim_url = nim_url
        self.nim_available = False
        
        if use_llm:
            self._check_nim()
    
    def _check_nim(self):
        """Check if NIM server is available."""
        print("[CommandParser] Connecting to NIM server...")
        self.nim_model = "meta/llama-3.1-8b-instruct"  # default
        
        try:
            response = requests.get(f"{self.nim_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                model_id = models.get("data", [{}])[0].get("id", "unknown")
                self.nim_model = model_id
                print(f"[CommandParser] NIM connected: {model_id}")
                self.nim_available = True
            else:
                raise Exception(f"NIM returned status {response.status_code}")
                
        except Exception as e:
            print(f"[CommandParser] WARNING: NIM not available: {e}")
            print("[CommandParser] Falling back to keyword matching")
            print("[CommandParser] Start NIM with: ./scripts/voice_assistant/nim_start.sh")
            self.use_llm = False
    
    def parse(self, text: str) -> VoiceCommand:
        """Parse spoken text into a command."""
        text = text.strip().lower()
        
        if not text:
            return VoiceCommand(command="unknown", raw_text=text)
        
        if self.use_llm and self.nim_available:
            return self._parse_with_llm(text)
        else:
            return self._parse_with_keywords(text)
    
    def _parse_with_llm(self, text: str) -> VoiceCommand:
        """Parse using Llama via NIM API."""
        try:
            response = requests.post(
                f"{self.nim_url}/v1/chat/completions",
                json={
                    "model": self.nim_model,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT},
                        {"role": "user", "content": text}
                    ],
                    "max_tokens": 100,
                    "temperature": 0.1,
                },
                timeout=10,
            )
            
            if response.status_code != 200:
                print(f"[CommandParser] NIM error: {response.status_code}")
                return self._parse_with_keywords(text)
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Extract JSON from response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = content[start:end]
                parsed = json.loads(json_str)
                
                command = parsed.get("command", "unknown")
                args = {k: v for k, v in parsed.items() if k != "command"}
                
                return VoiceCommand(command=command, args=args if args else None, raw_text=text)
                
        except requests.exceptions.Timeout:
            print("[CommandParser] NIM timeout, using keywords")
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"[CommandParser] NIM error: {e}")
        
        # Fall back to keyword matching if NIM fails
        return self._parse_with_keywords(text)
    
    def _parse_with_keywords(self, text: str) -> VoiceCommand:
        """Simple keyword-based parsing as fallback."""
        text = text.lower()
        
        # Recording commands
        if any(kw in text for kw in ["start record", "begin record", "start capturing"]):
            return VoiceCommand(command="start_recording", raw_text=text)
        
        if any(kw in text for kw in ["stop record", "end record", "stop capturing", "finish record"]):
            return VoiceCommand(command="stop_recording", raw_text=text)
        
        if text in ["stop", "done", "end"]:
            return VoiceCommand(command="stop_recording", raw_text=text)
        
        # Status queries
        if any(kw in text for kw in ["am i record", "are we record", "recording?", "status", 
                                      "what's the task", "what is the task", "current task",
                                      "what's the prompt", "what is the prompt"]):
            return VoiceCommand(command="get_status", raw_text=text)
        
        # Episode count (with optional date filter)
        if any(kw in text for kw in ["how many episode", "episode count", "number of episode",
                                      "total episode", "episodes recorded", "how many recording"]):
            args = {}
            if "today" in text:
                args["date_filter"] = "today"
            elif "yesterday" in text:
                args["date_filter"] = "yesterday"
            elif "this week" in text or "week" in text:
                args["date_filter"] = "this_week"
            return VoiceCommand(command="get_episode_count", args=args if args else None, raw_text=text)
        
        # Object commands
        if any(kw in text for kw in ["drop", "spawn", "place", "add", "give me", "get me"]):
            obj_type = None
            item = None
            
            # Check for specific fruits
            fruit_names = ["orange", "lemon", "lime", "avocado", "pomegranate", "lychee"]
            for fruit in fruit_names:
                if fruit in text:
                    obj_type = "fruits"
                    item = fruit
                    break
            
            # Check for general categories
            if not obj_type:
                if "cube" in text:
                    obj_type = "cubes"
                elif "mug" in text or "cup" in text:
                    obj_type = "mugs"
                elif "fruit" in text:
                    obj_type = "fruits"
            
            # Only send spawn command if we recognized a valid object
            if obj_type or item:
                args = {}
                if obj_type:
                    args["type"] = obj_type
                if item:
                    args["item"] = item
                return VoiceCommand(command="spawn_object", args=args, raw_text=text)
            else:
                # Unknown object - return error response instead of sending to teleop
                return VoiceCommand(command="invalid_object", raw_text=text)
        
        # Reset teleop (back to initial position) - check before reset_objects
        if any(kw in text for kw in ["reset teleop", "reset robot", "reset position", 
                                      "initial position", "restart teleop", "recalibrate"]):
            return VoiceCommand(command="reset_teleop", raw_text=text)
        
        # Reset objects (clear scene)
        if any(kw in text for kw in ["reset objects", "clear objects", "clear the table",
                                      "remove all", "clean up", "clear scene"]):
            return VoiceCommand(command="reset_objects", raw_text=text)
        
        # Generic "reset" or "clear" defaults to reset_objects for backward compatibility
        if text.strip() in ["reset", "clear"]:
            return VoiceCommand(command="reset_objects", raw_text=text)
        
        # Prompt/Label commands
        prompt_triggers = ["task is", "set the prompt to", "set prompt to", "set prompt", 
                          "the prompt is", "set task", "set label to", "label is", 
                          "the label is", "set the label to"]
        for trigger in prompt_triggers:
            if trigger in text:
                idx = text.find(trigger) + len(trigger)
                prompt_text = text[idx:].strip()
                if prompt_text:
                    return VoiceCommand(
                        command="set_prompt",
                        args={"prompt": prompt_text},
                        raw_text=text
                    )
        
        return VoiceCommand(command="unknown", raw_text=text)
    
    def chat(self, text: str) -> str:
        """Generate conversational response using NIM."""
        if not self.nim_available:
            return "I didn't catch that."
        
        try:
            response = requests.post(
                f"{self.nim_url}/v1/chat/completions",
                json={
                    "model": self.nim_model,
                    "messages": [
                        {"role": "system", "content": self.CHAT_PROMPT},
                        {"role": "user", "content": text}
                    ],
                    "max_tokens": 150,
                    "temperature": 0.7,
                },
                timeout=10,
            )
            
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["message"]["content"].strip()
                
        except Exception as e:
            print(f"[NIM] Chat error: {e}")
        
        return "I didn't catch that."


class RivaASR:
    """NVIDIA Riva streaming ASR client."""
    
    def __init__(self, server: str = "localhost:50051", sample_rate: int = 16000):
        self.server = server
        self.sample_rate = sample_rate
        self.channel = None
        self.stub = None
        
        self._connect()
    
    def _connect(self):
        """Connect to Riva server."""
        try:
            import riva.client
            
            self.auth = riva.client.Auth(uri=self.server)
            self.asr_service = riva.client.ASRService(self.auth)
            
            print(f"[RivaASR] Connected to Riva server at {self.server}")
            
        except ImportError:
            print("[RivaASR] ERROR: nvidia-riva-client not installed")
            print("  pip install nvidia-riva-client")
            raise
        except Exception as e:
            print(f"[RivaASR] ERROR: Could not connect to Riva: {e}")
            print("  Make sure Riva server is running: bash riva_start.sh")
            raise
    
    def transcribe_stream(self, audio_queue: queue.Queue, result_callback):
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
            while True:
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
                if not response.results:
                    continue
                
                for result in response.results:
                    if result.alternatives:
                        transcript = result.alternatives[0].transcript
                        is_final = result.is_final
                        
                        result_callback(transcript, is_final)
                        
        except Exception as e:
            print(f"[RivaASR] Stream error: {e}")


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
                print(f"[AudioCapture] {status}")
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
        print(f"[AudioCapture] Started (sample_rate={self.sample_rate})")
    
    def stop(self):
        """Stop audio capture."""
        self.running = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.audio_queue.put(None)
        print("[AudioCapture] Stopped")
    
    def get_queue(self) -> queue.Queue:
        return self.audio_queue


class RivaTTS:
    """NVIDIA Riva Text-to-Speech with auto-detection of Riva version."""
    
    def __init__(self, server: str = "localhost:50051"):
        self.server = server
        self.sample_rate = 22050
        self.tts_service = None
        self.voice_name = "English-US.Male-1"  # Default for Riva 2.18
        
        try:
            import riva.client
            
            self.auth = riva.client.Auth(uri=server)
            self.tts_service = riva.client.SpeechSynthesisService(self.auth)
            
            # Auto-detect Riva version by checking available voices
            self._detect_voice()
            print(f"[TTS] Connected to Riva TTS, using voice: {self.voice_name}")
            
        except Exception as e:
            print(f"[TTS] Could not connect: {e}")
    
    def _detect_voice(self):
        """Detect available voices and select appropriate one."""
        try:
            # Try Magpie voice first (Riva 2.19)
            import riva.client
            test_voices = [
                "Magpie-Multilingual.EN-US.Male.Male-1",  # Riva 2.19 Magpie
                "English-US.Male-1",  # Riva 2.18 FastPitch
            ]
            # We'll try the first one and fall back if it fails
            # For now, check Docker to determine version
            import subprocess
            result = subprocess.run(
                ["docker", "ps", "--format", "{{.Image}}"],
                capture_output=True, text=True, timeout=5
            )
            if "2.19" in result.stdout:
                self.voice_name = "Magpie-Multilingual.EN-US.Male.Male-1"
                print("[TTS] Detected Riva 2.19 (Magpie)")
            else:
                self.voice_name = "English-US.Male-1"
                print("[TTS] Detected Riva 2.18 (FastPitch)")
        except Exception as e:
            print(f"[TTS] Voice detection failed, using default: {e}")
            self.voice_name = "English-US.Male-1"
    
    def speak(self, text: str):
        """Synthesize and play speech."""
        if not self.tts_service or not text.strip():
            return
        
        try:
            import riva.client
            import sounddevice as sd
            import numpy as np
            
            responses = self.tts_service.synthesize_online(
                text,
                voice_name=self.voice_name,
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


class VoiceServer:
    """Main voice assistant server."""
    
    def __init__(
        self,
        riva_server: str = "localhost:50051",
        tts_server: str = None,
        zmq_port: int = 5556,
        llm_device: str = "cuda:0",
        use_llm: bool = True,
        enable_tts: bool = True,
        nim_url: str = "http://localhost:8000",
    ):
        self.zmq_port = zmq_port
        self.running = False
        self.enable_tts = enable_tts
        if tts_server is None:
            tts_server = riva_server
        
        # Initialize components
        print("\n" + "=" * 50)
        print("JAX - SPARK PACK VOICE ASSISTANT")
        print("=" * 50)
        
        print("\n[1/5] Initializing ZMQ publisher...")
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.PUB)
        self.zmq_socket.bind(f"tcp://*:{zmq_port}")
        print(f"  Publishing commands on tcp://*:{zmq_port}")
        
        # Subscribe to responses from teleop
        self.response_socket = self.zmq_context.socket(zmq.SUB)
        self.response_socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.response_socket.setsockopt(zmq.RCVTIMEO, 0)  # Non-blocking
        self.response_socket.connect("tcp://localhost:5557")
        print(f"  Listening for responses on tcp://localhost:5557")
        
        print("\n[2/5] Initializing command parser (NIM)...")
        self.parser = CommandParser(device=llm_device, use_llm=use_llm, nim_url=nim_url)
        
        print("\n[3/5] Initializing Riva ASR...")
        self.asr = RivaASR(server=riva_server)
        
        print("\n[4/5] Initializing TTS...")
        self.tts = RivaTTS(server=tts_server) if enable_tts else None
        
        print("\n[5/5] Initializing audio capture...")
        self.audio = AudioCapture()
        
        # State
        self.current_transcript = ""
        self.last_final_time = 0
        self.silence_threshold = 1.5  # seconds of silence before processing
        
        # Teleop connection tracking
        self.last_teleop_response_time = 0
        self.teleop_connected = False
        self.pending_command_time = 0  # When we sent a command expecting response
        
        print("\n" + "=" * 50)
        print("JAX READY - Listening for voice commands")
        print("=" * 50)
        print("\nAvailable commands:")
        print("  - 'Start recording' / 'Stop recording'")
        print("  - 'Drop a lemon' / 'Spawn an orange'")
        print("  - 'Drop a cube' / 'Spawn a mug'")
        print("  - 'Reset objects' / 'Clear the table'")
        print("  - 'Reset teleop' / 'Reset robot' (back to initial position)")
        print("  - 'Set label to pick up the lemon' / 'The task is sort fruit'")
        print("  - 'Status' / 'Am I recording?'")
        print("  - 'How many episodes?' / 'Episodes recorded today/yesterday'")
        print("\nAvailable fruits: orange, lemon, lime, avocado, pomegranate, lychee")
        print("\nPress Ctrl+C to quit\n")
    
    def _on_transcript(self, text: str, is_final: bool):
        """Handle transcription results."""
        if is_final:
            self.current_transcript = text
            self.last_final_time = time.time()
            
            # Process command
            cmd = self.parser.parse(text)
            
            if cmd.command != "unknown":
                print(f"\n[VOICE] '{text}'")
                print(f"[CMD] {cmd.command}", end="")
                if cmd.args:
                    print(f" {cmd.args}", end="")
                print()
                
                # Handle invalid object command locally (don't send to teleop)
                if cmd.command == "invalid_object":
                    response = "I don't recognize that object. Available: orange, lemon, lime, avocado, pomegranate, lychee, cube, or mug."
                    print(f"[JAX] {response}")
                    if self.tts:
                        self.tts.speak(response)
                # Check if this is a robot command
                elif self._is_robot_command(cmd.command):
                    # Send via ZMQ and track that we're waiting for response
                    self.zmq_socket.send_json(cmd.to_json())
                    self.pending_command_time = time.time()
                    print(f"[ZMQ] Sent: {cmd.to_json()}")
                else:
                    # Non-robot command (e.g., conversation)
                    self.zmq_socket.send_json(cmd.to_json())
                    print(f"[ZMQ] Sent: {cmd.to_json()}")
            else:
                print(f"[VOICE] '{text}' (not a command, using chat)")
                # Use chat mode for conversational responses
                response = self.parser.chat(text)
                print(f"[JAX] {response}")
                if self.tts:
                    self.tts.speak(response)
        else:
            # Interim result - show what's being said
            if text.strip():
                print(f"\r[...] {text}        ", end="", flush=True)
    
    def _check_responses(self):
        """Check for responses from teleop and speak them."""
        try:
            msg = self.response_socket.recv_json(zmq.NOBLOCK)
            response = msg.get("response", "")
            
            # Update connection tracking
            was_connected = self.teleop_connected
            self.last_teleop_response_time = time.time()
            self.teleop_connected = True
            self.pending_command_time = 0  # Clear pending
            
            if response:
                # Actual response - print and speak
                print(f"[TELEOP] {response}")
                if self.tts:
                    self.tts.speak(response)
            elif not was_connected:
                # First connection (heartbeat after being disconnected)
                print("[TELEOP] Client connected")
                
        except zmq.Again:
            # Check if we're waiting for a response that hasn't come
            if self.pending_command_time > 0:
                elapsed = time.time() - self.pending_command_time
                if elapsed > 2.0:  # 2 second timeout
                    self.teleop_connected = False
                    self.pending_command_time = 0
                    print("[WARN] Teleop not responding")
                    if self.tts:
                        self.tts.speak("Teleop client is not running. Please start it first.")
            
            # Check for connection timeout (no heartbeat in 10 seconds)
            if self.teleop_connected and self.last_teleop_response_time > 0:
                if time.time() - self.last_teleop_response_time > 10.0:
                    self.teleop_connected = False
                    print("[WARN] Lost connection to teleop")
                    
        except Exception as e:
            pass  # Ignore errors
    
    def _is_robot_command(self, command: str) -> bool:
        """Check if command requires teleop to be running."""
        return command in ["start_recording", "stop_recording", "spawn_object", 
                          "reset_objects", "reset_teleop", "set_prompt", "get_status",
                          "get_episode_count"]
    
    def run(self):
        """Run the voice server."""
        self.running = True
        
        # Greeting
        if self.tts:
            self.tts.speak("JAX online. Ready to assist with Spark Pack operations.")
        
        # Start audio capture
        self.audio.start()
        
        # Start ASR in separate thread
        asr_thread = threading.Thread(
            target=self.asr.transcribe_stream,
            args=(self.audio.get_queue(), self._on_transcript),
            daemon=True,
        )
        asr_thread.start()
        
        try:
            while self.running:
                # Check for responses from teleop
                self._check_responses()
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\n\n[INFO] Shutting down...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop the voice server."""
        self.running = False
        self.audio.stop()
        if self.tts:
            self.tts.speak("JAX signing off.")
        self.zmq_socket.close()
        self.response_socket.close()
        self.zmq_context.term()
        print("[INFO] JAX stopped")


def main():
    parser = argparse.ArgumentParser(description="JAX - Spark Pack Voice Assistant")
    parser.add_argument("--riva-server", type=str, default="localhost:50051",
                        help="Riva ASR gRPC server address")
    parser.add_argument("--tts-server", type=str, default=None,
                        help="Riva TTS gRPC server address (defaults to --riva-server)")
    parser.add_argument("--zmq-port", type=int, default=5556,
                        help="ZMQ publisher port")
    parser.add_argument("--llm-device", type=str, default="cuda:0",
                        help="Device for Llama model (unused with NIM)")
    parser.add_argument("--nim-url", type=str, default="http://localhost:8000",
                        help="NIM server URL for Llama inference")
    parser.add_argument("--no-llm", action="store_true",
                        help="Disable LLM, use keyword matching only")
    parser.add_argument("--no-tts", action="store_true",
                        help="Disable text-to-speech output")
    args = parser.parse_args()
    
    server = VoiceServer(
        riva_server=args.riva_server,
        tts_server=args.tts_server,
        zmq_port=args.zmq_port,
        llm_device=args.llm_device,
        nim_url=args.nim_url,
        enable_tts=not args.no_tts,
        use_llm=not args.no_llm,
    )
    server.run()


if __name__ == "__main__":
    main()
