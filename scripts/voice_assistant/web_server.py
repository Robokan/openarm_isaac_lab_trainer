#!/usr/bin/env python3
"""
JAX Web Interface Server

Provides a web interface for talking to JAX from any device (iPhone, tablet, etc.)
Uses WebSocket for real-time audio streaming.

Usage:
    python web_server.py
    python web_server.py --port 8080

Then open http://<your-ip>:8080 on your iPhone
"""

import argparse
import asyncio
import json
import os
import struct
import threading
import time
from pathlib import Path

import numpy as np
import requests

# Web server imports
try:
    from aiohttp import web
    import aiohttp
except ImportError:
    print("Installing aiohttp...")
    import subprocess
    subprocess.run(["pip", "install", "aiohttp"])
    from aiohttp import web
    import aiohttp

import zmq


# System prompt for NIM LLM parsing
NIM_SYSTEM_PROMPT = """You are JAX, a voice command parser for Spark Pack's robot teleoperation system.
Given a spoken command, extract the intent and return a JSON object.

Valid commands:
- start_recording: User wants to start recording a demonstration
- stop_recording: User wants to stop recording
- get_status: User is asking about current state (recording status, task, etc.)
- spawn_object: User wants to drop/spawn an object
  - type: "cubes", "mugs", or "fruits" (category)
  - item: specific item name (e.g., "lemon", "orange", "cube", "mug")
- reset_objects: User wants to clear/reset all objects
- set_prompt: User wants to set the task description (extract the task text)
- unknown: Command not recognized

Available items:
- Fruits: orange, lemon, lime, avocado, pomegranate, lychee
- Cubes: colored cubes
- Mugs/Cups: coffee mugs

Examples:
User: "start recording" -> {"command": "start_recording"}
User: "drop a lemon" -> {"command": "spawn_object", "type": "fruits", "item": "lemon"}
User: "status" -> {"command": "get_status"}
User: "the task is pick up the apple" -> {"command": "set_prompt", "prompt": "pick up the apple"}

Return ONLY the JSON object, no other text."""

# System prompt for conversational chat mode
CHAT_SYSTEM_PROMPT = """You are JAX, an advanced AI assistant for Spark Pack's robotics laboratory. Think of yourself as Jarvis - sophisticated, composed, and effortlessly competent.

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


class JAXWebServer:
    """Web server for JAX voice interface."""
    
    def __init__(self, port: int = 8080, riva_server: str = "localhost:50051", tts_server: str = None, nim_url: str = "http://localhost:8000"):
        self.port = port
        self.riva_server = riva_server
        self.tts_server = tts_server if tts_server else riva_server
        self.nim_url = nim_url
        self.nim_available = False
        self.app = web.Application()
        self.clients = set()
        
        # Check NIM availability
        self._check_nim()
        
        # ZMQ for talking to teleop
        self.zmq_context = zmq.Context()
        self.zmq_pub = self.zmq_context.socket(zmq.PUB)
        self.zmq_pub.bind("tcp://*:5556")
        
        self.zmq_sub = self.zmq_context.socket(zmq.SUB)
        self.zmq_sub.setsockopt(zmq.SUBSCRIBE, b"")
        self.zmq_sub.setsockopt(zmq.RCVTIMEO, 100)
        self.zmq_sub.connect("tcp://localhost:5557")
        
        # Riva clients
        self.asr_service = None
        self.tts_service = None
        self._init_riva()
        
        # Setup routes
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_get('/ws', self.handle_websocket)
        
    def _check_nim(self):
        """Check if NIM server is available."""
        self.nim_model = "meta/llama-3.1-8b-instruct"  # default
        try:
            response = requests.get(f"{self.nim_url}/v1/models", timeout=5)
            if response.status_code == 200:
                models = response.json()
                model_id = models.get("data", [{}])[0].get("id", "unknown")
                self.nim_model = model_id
                print(f"[NIM] Connected to {self.nim_url} ({model_id})")
                self.nim_available = True
            else:
                raise Exception(f"Status {response.status_code}")
        except Exception as e:
            print(f"[NIM] Not available: {e}")
            print("[NIM] Using keyword matching for commands")
    
    def _init_riva(self):
        """Initialize Riva ASR and TTS."""
        try:
            import riva.client
            
            auth_asr = riva.client.Auth(uri=self.riva_server)
            self.asr_service = riva.client.ASRService(auth_asr)
            print(f"[Riva] ASR connected to {self.riva_server}")
            auth_tts = riva.client.Auth(uri=self.tts_server)
            self.tts_service = riva.client.SpeechSynthesisService(auth_tts)
            print(f"[Riva] TTS connected to {self.tts_server}")
        except Exception as e:
            print(f"[Riva] Could not connect: {e}")
            print("[Riva] ASR/TTS will not work until Riva is running")
    
    async def handle_index(self, request):
        """Serve the main HTML page."""
        html = self._get_html()
        return web.Response(text=html, content_type='text/html')
    
    async def handle_websocket(self, request):
        """Handle WebSocket connections for audio streaming."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.clients.add(ws)
        print(f"[WS] Client connected ({len(self.clients)} total)")
        
        # Send welcome message
        await ws.send_json({"type": "status", "message": "Connected to JAX"})
        
        try:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    await self._handle_message(ws, data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    # Audio data from client
                    await self._handle_audio(ws, msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    print(f"[WS] Error: {ws.exception()}")
        finally:
            self.clients.discard(ws)
            print(f"[WS] Client disconnected ({len(self.clients)} total)")
        
        return ws
    
    async def _handle_message(self, ws, data):
        """Handle JSON messages from client."""
        msg_type = data.get("type", "")
        
        if msg_type == "transcribe":
            # Client sent text to process (fallback if no ASR)
            text = data.get("text", "")
            await self._process_text(ws, text)
            
        elif msg_type == "command":
            # Direct command
            cmd = data.get("command", {})
            self.zmq_pub.send_json(cmd)
            await ws.send_json({"type": "status", "message": f"Sent: {cmd}"})
    
    async def _handle_audio(self, ws, audio_data):
        """Handle audio data from client - transcribe with Riva."""
        if not self.asr_service:
            await ws.send_json({"type": "error", "message": "Riva ASR not available"})
            return
        
        try:
            import riva.client
            
            # Transcribe audio
            config = riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=16000,
                language_code="en-US",
                max_alternatives=1,
                enable_automatic_punctuation=True,
            )
            
            response = self.asr_service.offline_recognize(audio_data, config)
            
            if response.results:
                text = response.results[0].alternatives[0].transcript
                if text.strip():
                    await ws.send_json({"type": "transcript", "text": text})
                    await self._process_text(ws, text)
                    
        except Exception as e:
            print(f"[ASR] Error: {e}")
            await ws.send_json({"type": "error", "message": str(e)})
    
    async def _process_text(self, ws, text: str):
        """Process transcribed text - parse command and execute."""
        text = text.strip().lower()
        if not text:
            return
        
        # Parse command (simple keyword matching for web interface)
        cmd = self._parse_command(text)
        
        if cmd["command"] != "unknown":
            # Send to teleop
            self.zmq_pub.send_json(cmd)
            await ws.send_json({"type": "command", "command": cmd})
            
            # Wait for response
            response = await self._wait_for_response()
            if response:
                await ws.send_json({"type": "response", "message": response})
                await self._speak(ws, response)
            else:
                msg = "Teleop client is not responding"
                await ws.send_json({"type": "response", "message": msg})
                await self._speak(ws, msg)
        else:
            # Try conversational response if NIM is available
            if self.nim_available:
                msg = self._chat_with_nim(text)
            else:
                msg = "I didn't understand that. Try: start recording, drop a lemon, status"
            await ws.send_json({"type": "response", "message": msg})
            await self._speak(ws, msg)
    
    def _chat_with_nim(self, text: str) -> str:
        """Generate conversational response using NIM."""
        try:
            response = requests.post(
                f"{self.nim_url}/v1/chat/completions",
                json={
                    "model": self.nim_model,
                    "messages": [
                        {"role": "system", "content": CHAT_SYSTEM_PROMPT},
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
        
        return "I didn't understand that. Try: start recording, drop a lemon, status"
    
    def _parse_command(self, text: str) -> dict:
        """Parse text into command using NIM or keyword fallback."""
        if self.nim_available:
            result = self._parse_with_nim(text)
            if result:
                return result
        return self._parse_with_keywords(text)
    
    def _parse_with_nim(self, text: str) -> dict:
        """Parse using NIM LLM."""
        try:
            response = requests.post(
                f"{self.nim_url}/v1/chat/completions",
                json={
                    "model": self.nim_model,
                    "messages": [
                        {"role": "system", "content": NIM_SYSTEM_PROMPT},
                        {"role": "user", "content": text}
                    ],
                    "max_tokens": 100,
                    "temperature": 0.1,
                },
                timeout=10,
            )
            
            if response.status_code != 200:
                return None
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Extract JSON from response
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
                
        except Exception as e:
            print(f"[NIM] Parse error: {e}")
        
        return None
    
    def _parse_with_keywords(self, text: str) -> dict:
        """Simple keyword-based parsing as fallback."""
        # Recording
        if any(kw in text for kw in ["start record", "begin record"]):
            return {"command": "start_recording"}
        if any(kw in text for kw in ["stop record", "end record", "stop", "done"]):
            return {"command": "stop_recording"}
        
        # Status
        if any(kw in text for kw in ["status", "am i record", "are we record"]):
            return {"command": "get_status"}
        
        # Objects
        if any(kw in text for kw in ["drop", "spawn", "give me"]):
            # Check for specific fruits
            fruits = ["orange", "lemon", "lime", "avocado", "pomegranate", "lychee"]
            for fruit in fruits:
                if fruit in text:
                    return {"command": "spawn_object", "type": "fruits", "item": fruit}
            if "cube" in text:
                return {"command": "spawn_object", "type": "cubes"}
            if "mug" in text or "cup" in text:
                return {"command": "spawn_object", "type": "mugs"}
            return {"command": "spawn_object"}
        
        if any(kw in text for kw in ["reset", "clear"]):
            return {"command": "reset_objects"}
        
        # Prompt
        for trigger in ["task is", "set prompt", "prompt is"]:
            if trigger in text:
                idx = text.find(trigger) + len(trigger)
                prompt = text[idx:].strip()
                if prompt:
                    return {"command": "set_prompt", "prompt": prompt}
        
        return {"command": "unknown"}
    
    async def _wait_for_response(self, timeout: float = 2.0) -> str:
        """Wait for response from teleop."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                msg = self.zmq_sub.recv_json(zmq.NOBLOCK)
                response = msg.get("response", "")
                if response:
                    return response
            except zmq.Again:
                pass
            await asyncio.sleep(0.05)
        return ""
    
    async def _speak(self, ws, text: str):
        """Generate TTS audio and send to client."""
        if not self.tts_service or not text:
            return
        
        try:
            import riva.client
            
            responses = self.tts_service.synthesize_online(
                text,
                voice_name="Magpie-Multilingual.EN-US.Male.Calm",
                language_code="en-US",
                sample_rate_hz=22050,
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
            )
            
            audio_chunks = []
            for response in responses:
                audio_chunks.append(response.audio)
            
            if audio_chunks:
                audio_data = b''.join(audio_chunks)
                # Send as base64 for easier handling in JS
                import base64
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')
                await ws.send_json({
                    "type": "audio",
                    "data": audio_b64,
                    "sample_rate": 22050
                })
                
        except Exception as e:
            print(f"[TTS] Error: {e}")
    
    def _get_html(self) -> str:
        """Return the HTML page for the web interface."""
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>JAX - Spark Pack</title>
    <style>
        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        .container {
            max-width: 400px;
            width: 100%;
            text-align: center;
        }
        h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .subtitle {
            color: #888;
            margin-bottom: 30px;
        }
        .status {
            background: rgba(255,255,255,0.1);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .status.connected {
            border-left: 4px solid #00ff88;
        }
        .status.disconnected {
            border-left: 4px solid #ff4444;
        }
        .mic-button {
            width: 120px;
            height: 120px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #7c3aed, #00d4ff);
            color: white;
            font-size: 40px;
            cursor: pointer;
            margin: 30px 0;
            transition: transform 0.2s, box-shadow 0.2s;
            box-shadow: 0 10px 30px rgba(124, 58, 237, 0.3);
        }
        .mic-button:active, .mic-button.recording {
            transform: scale(0.95);
            background: linear-gradient(135deg, #ff4444, #ff8800);
            box-shadow: 0 5px 20px rgba(255, 68, 68, 0.5);
        }
        .mic-button.recording {
            animation: pulse 1s infinite;
        }
        @keyframes pulse {
            0%, 100% { transform: scale(0.95); }
            50% { transform: scale(1.0); }
        }
        .transcript {
            background: rgba(255,255,255,0.05);
            padding: 20px;
            border-radius: 10px;
            min-height: 100px;
            margin-bottom: 20px;
            text-align: left;
        }
        .transcript .you {
            color: #00d4ff;
            margin-bottom: 10px;
        }
        .transcript .jax {
            color: #00ff88;
        }
        .quick-commands {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 10px;
            margin-top: 20px;
        }
        .quick-btn {
            padding: 15px;
            border: none;
            border-radius: 10px;
            background: rgba(255,255,255,0.1);
            color: white;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .quick-btn:active {
            background: rgba(255,255,255,0.2);
        }
        .text-input-container {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        .text-input {
            flex: 1;
            padding: 15px;
            border: none;
            border-radius: 10px;
            background: rgba(255,255,255,0.1);
            color: white;
            font-size: 16px;
            outline: none;
        }
        .text-input::placeholder {
            color: #666;
        }
        .text-input:focus {
            background: rgba(255,255,255,0.15);
            box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.5);
        }
        .send-btn {
            padding: 15px 25px;
            border: none;
            border-radius: 10px;
            background: linear-gradient(135deg, #7c3aed, #00d4ff);
            color: white;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .send-btn:active {
            transform: scale(0.95);
        }
        .instructions {
            margin-top: 30px;
            padding: 15px;
            background: rgba(255,255,255,0.05);
            border-radius: 10px;
            font-size: 12px;
            color: #888;
            text-align: left;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>JAX</h1>
        <p class="subtitle">Spark Pack Voice Assistant</p>
        
        <div id="status" class="status disconnected">
            Connecting...
        </div>
        
        <button id="micBtn" class="mic-button">🎤</button>
        <p id="micStatus">Tap to speak</p>
        
        <div class="transcript">
            <div id="youSaid" class="you"></div>
            <div id="jaxSaid" class="jax"></div>
        </div>
        
        <div class="text-input-container">
            <input type="text" id="textInput" class="text-input" placeholder="Type a command..." 
                   onkeypress="if(event.key==='Enter')sendTypedText()">
            <button class="send-btn" onclick="sendTypedText()">Send</button>
        </div>
        
        <div class="quick-commands">
            <button class="quick-btn" onclick="sendCommand('start_recording')">▶️ Start Recording</button>
            <button class="quick-btn" onclick="sendCommand('stop_recording')">⏹️ Stop Recording</button>
            <button class="quick-btn" onclick="sendCommand('spawn_object')">📦 Drop Object</button>
            <button class="quick-btn" onclick="sendCommand('reset_objects')">🗑️ Reset</button>
            <button class="quick-btn" onclick="sendCommand('get_status')">ℹ️ Status</button>
            <button class="quick-btn" onclick="sendText('drop a lemon')">🍋 Lemon</button>
        </div>
        
        <div class="instructions">
            <strong>Commands (voice or text):</strong><br>
            • "Start recording" / "Stop"<br>
            • "Drop a lemon" / "Spawn a cube"<br>
            • "Reset objects" / "Status"<br>
            • "Set the task to [description]"
        </div>
    </div>
    
    <script>
        let ws = null;
        let isRecording = false;
        
        const statusEl = document.getElementById('status');
        const micBtn = document.getElementById('micBtn');
        const micStatus = document.getElementById('micStatus');
        const youSaid = document.getElementById('youSaid');
        const jaxSaid = document.getElementById('jaxSaid');
        
        // Connect WebSocket
        function connect() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${location.host}/ws`);
            
            ws.onopen = () => {
                statusEl.textContent = 'Connected to JAX';
                statusEl.className = 'status connected';
            };
            
            ws.onclose = () => {
                statusEl.textContent = 'Disconnected - Reconnecting...';
                statusEl.className = 'status disconnected';
                setTimeout(connect, 2000);
            };
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                handleMessage(data);
            };
        }
        
        function handleMessage(data) {
            switch(data.type) {
                case 'status':
                    statusEl.textContent = data.message;
                    break;
                case 'transcript':
                    youSaid.textContent = 'You: ' + data.text;
                    break;
                case 'response':
                    jaxSaid.textContent = 'JAX: ' + data.message;
                    break;
                case 'audio':
                    playAudio(data.data, data.sample_rate);
                    break;
                case 'error':
                    jaxSaid.textContent = 'Error: ' + data.message;
                    break;
            }
        }
        
        function playAudio(base64Data, sampleRate) {
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const binaryString = atob(base64Data);
            const len = binaryString.length;
            const bytes = new Int16Array(len / 2);
            
            for (let i = 0; i < len; i += 2) {
                bytes[i/2] = binaryString.charCodeAt(i) | (binaryString.charCodeAt(i+1) << 8);
            }
            
            const floatData = new Float32Array(bytes.length);
            for (let i = 0; i < bytes.length; i++) {
                floatData[i] = bytes[i] / 32768.0;
            }
            
            const audioBuffer = audioContext.createBuffer(1, floatData.length, sampleRate);
            audioBuffer.getChannelData(0).set(floatData);
            
            const source = audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(audioContext.destination);
            source.start();
        }
        
        // Microphone handling with raw PCM capture
        let audioContext = null;
        let audioStream = null;
        let scriptProcessor = null;
        let audioData = [];
        
        async function startRecording() {
            try {
                audioStream = await navigator.mediaDevices.getUserMedia({ 
                    audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true } 
                });
                
                audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
                const source = audioContext.createMediaStreamSource(audioStream);
                
                // Use ScriptProcessorNode to capture raw PCM
                scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);
                audioData = [];
                
                scriptProcessor.onaudioprocess = (e) => {
                    const inputData = e.inputBuffer.getChannelData(0);
                    // Convert float32 to int16
                    const int16Data = new Int16Array(inputData.length);
                    for (let i = 0; i < inputData.length; i++) {
                        int16Data[i] = Math.max(-32768, Math.min(32767, inputData[i] * 32768));
                    }
                    audioData.push(new Int16Array(int16Data));
                };
                
                source.connect(scriptProcessor);
                scriptProcessor.connect(audioContext.destination);
                
                isRecording = true;
                micBtn.classList.add('recording');
                micStatus.textContent = 'Listening... tap to stop';
                
            } catch (err) {
                alert('Could not access microphone: ' + err.message);
            }
        }
        
        function stopRecording() {
            if (isRecording) {
                isRecording = false;
                micBtn.classList.remove('recording');
                micStatus.textContent = 'Processing...';
                
                // Stop audio processing
                if (scriptProcessor) {
                    scriptProcessor.disconnect();
                    scriptProcessor = null;
                }
                if (audioStream) {
                    audioStream.getTracks().forEach(track => track.stop());
                    audioStream = null;
                }
                
                // Combine all audio chunks into single buffer
                const totalLength = audioData.reduce((acc, chunk) => acc + chunk.length, 0);
                const combined = new Int16Array(totalLength);
                let offset = 0;
                for (const chunk of audioData) {
                    combined.set(chunk, offset);
                    offset += chunk.length;
                }
                
                // Send raw PCM to server
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(combined.buffer);
                }
                
                audioData = [];
                micStatus.textContent = 'Tap to speak';
                
                if (audioContext) {
                    audioContext.close();
                    audioContext = null;
                }
            }
        }
        
        micBtn.addEventListener('click', () => {
            if (isRecording) {
                stopRecording();
            } else {
                startRecording();
            }
        });
        
        // Quick command buttons
        function sendCommand(command) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'command', command: { command: command } }));
                youSaid.textContent = 'Command: ' + command;
            }
        }
        
        function sendText(text) {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'transcribe', text: text }));
                youSaid.textContent = 'You: ' + text;
            }
        }
        
        function sendTypedText() {
            const input = document.getElementById('textInput');
            const text = input.value.trim();
            if (text && ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'transcribe', text: text }));
                youSaid.textContent = 'You: ' + text;
                input.value = '';
            }
        }
        
        // Start connection
        connect();
    </script>
</body>
</html>'''
    
    def run(self):
        """Run the web server."""
        print(f"\n{'='*50}")
        print("JAX WEB INTERFACE")
        print(f"{'='*50}")
        print(f"\nOpen in browser: http://localhost:{self.port}")
        print(f"Or from iPhone:  http://<your-ip>:{self.port}")
        print(f"\nTo find your IP: hostname -I")
        print(f"{'='*50}\n")
        
        web.run_app(self.app, port=self.port, print=None)


def main():
    parser = argparse.ArgumentParser(description="JAX Web Interface")
    parser.add_argument("--port", type=int, default=8080, help="Web server port")
    parser.add_argument("--riva-server", type=str, default="localhost:50051",
                        help="Riva ASR gRPC server address")
    parser.add_argument("--tts-server", type=str, default=None,
                        help="Riva TTS gRPC server address (defaults to --riva-server)")
    parser.add_argument("--nim-url", type=str, default="http://localhost:8000",
                        help="NIM server URL for Llama inference")
    args = parser.parse_args()
    
    server = JAXWebServer(port=args.port, riva_server=args.riva_server, tts_server=args.tts_server, nim_url=args.nim_url)
    server.run()


if __name__ == "__main__":
    main()
