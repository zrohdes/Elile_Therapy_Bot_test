import streamlit as st
import asyncio
import base64
import datetime
import os
import threading
import queue
import time
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from hume.client import AsyncHumeClient
from hume.empathic_voice.chat.socket_client import ChatConnectOptions
from hume.empathic_voice.chat.types import SubscribeEvent
from hume import MicrophoneInterface, Stream
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Hume Voice Chat",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


class StreamlitWebSocketHandler:
    def __init__(self):
        self.byte_strs = Stream.new()
        self.messages = []
        self.emotion_scores = {}
        self.chat_metadata = {}
        self.is_connected = False
        self.error_message = None

    async def on_open(self):
        self.is_connected = True
        self._log_message("WebSocket connection opened.", "system")

    async def on_message(self, message: SubscribeEvent):
        try:
            if message.type == "chat_metadata":
                self.chat_metadata = {
                    "chat_id": message.chat_id,
                    "chat_group_id": message.chat_group_id
                }
                self._log_message(f"Chat ID: {message.chat_id}", "metadata")
                return

            elif message.type == "user_message" or message.type == "assistant_message":
                msg_content = {
                    "role": message.message.role,
                    "content": message.message.content,
                    "timestamp": datetime.datetime.now(tz=datetime.timezone.utc),
                    "emotions": None
                }

                if message.models.prosody is not None:
                    emotions = self._extract_top_n_emotions(
                        dict(message.models.prosody.scores), 5
                    )
                    msg_content["emotions"] = emotions
                    self.emotion_scores = emotions

                self.messages.append(msg_content)
                return

            elif message.type == "audio_output":
                await self.byte_strs.put(
                    base64.b64decode(message.data.encode("utf-8"))
                )
                return

            elif message.type == "error":
                self.error_message = f"Hume API Error ({message.code}): {message.message}"
                self._log_message(self.error_message, "error")
                return

            else:
                self._log_message(f"Received {message.type} event", "system")

        except Exception as e:
            self.error_message = f"Handler error: {str(e)}"
            self._log_message(self.error_message, "error")

    async def on_close(self):
        self.is_connected = False
        self._log_message("WebSocket connection closed.", "system")

    async def on_error(self, error):
        self.error_message = f"WebSocket error: {str(error)}"
        self._log_message(self.error_message, "error")

    def _log_message(self, text: str, msg_type: str) -> None:
        timestamp = datetime.datetime.now(tz=datetime.timezone.utc).strftime("%H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "message": text,
            "type": msg_type
        }
        if not hasattr(self, 'logs'):
            self.logs = []
        self.logs.append(log_entry)

    def _extract_top_n_emotions(self, emotion_scores: dict, n: int) -> dict:
        sorted_emotions = sorted(emotion_scores.items(), key=lambda item: item[1], reverse=True)
        return {emotion: score for emotion, score in sorted_emotions[:n]}


class HumeVoiceChatApp:
    def __init__(self):
        self.handler = None
        self.client = None
        self.chat_task = None
        self.is_running = False

    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'handler' not in st.session_state:
            st.session_state.handler = StreamlitWebSocketHandler()
        if 'is_connected' not in st.session_state:
            st.session_state.is_connected = False
        if 'chat_started' not in st.session_state:
            st.session_state.chat_started = False

    def render_sidebar(self):
        """Render the sidebar with configuration and controls."""
        st.sidebar.title("🎙️ Hume Voice Chat")
        st.sidebar.markdown("---")

        # API Configuration
        st.sidebar.subheader("API Configuration")

        # Load environment variables
        load_dotenv()

        api_key = st.sidebar.text_input(
            "Hume API Key",
            value=os.getenv("HUME_API_KEY", ""),
            type="password",
            help="Your Hume API key"
        )

        secret_key = st.sidebar.text_input(
            "Hume Secret Key",
            value=os.getenv("HUME_SECRET_KEY", ""),
            type="password",
            help="Your Hume secret key"
        )

        config_id = st.sidebar.text_input(
            "Hume Config ID",
            value=os.getenv("HUME_CONFIG_ID", ""),
            help="Your Hume configuration ID"
        )

        st.sidebar.markdown("---")

        # Connection Controls
        st.sidebar.subheader("Connection Controls")

        if not st.session_state.chat_started:
            if st.sidebar.button("🚀 Start Voice Chat", type="primary"):
                if api_key and secret_key and config_id:
                    self.start_chat(api_key, secret_key, config_id)
                else:
                    st.sidebar.error("Please provide all required API credentials")
        else:
            if st.sidebar.button("🛑 Stop Voice Chat", type="secondary"):
                self.stop_chat()

        # Connection Status
        st.sidebar.subheader("Connection Status")
        if st.session_state.is_connected:
            st.sidebar.success("🟢 Connected")
        elif st.session_state.chat_started:
            st.sidebar.warning("🟡 Connecting...")
        else:
            st.sidebar.error("🔴 Disconnected")

        # Chat Metadata
        if hasattr(st.session_state.handler, 'chat_metadata') and st.session_state.handler.chat_metadata:
            st.sidebar.subheader("Chat Info")
            metadata = st.session_state.handler.chat_metadata
            st.sidebar.text(f"Chat ID: {metadata.get('chat_id', 'N/A')}")
            st.sidebar.text(f"Group ID: {metadata.get('chat_group_id', 'N/A')}")

    def start_chat(self, api_key: str, secret_key: str, config_id: str):
        """Start the voice chat session."""
        try:
            st.session_state.chat_started = True
            st.session_state.handler = StreamlitWebSocketHandler()

            # Start the chat in a separate thread
            def run_chat():
                asyncio.run(self._chat_loop(api_key, secret_key, config_id))

            chat_thread = threading.Thread(target=run_chat, daemon=True)
            chat_thread.start()

            st.sidebar.success("Voice chat started! Please allow microphone access.")

        except Exception as e:
            st.sidebar.error(f"Failed to start chat: {str(e)}")
            st.session_state.chat_started = False

    def stop_chat(self):
        """Stop the voice chat session."""
        st.session_state.chat_started = False
        st.session_state.is_connected = False
        st.sidebar.info("Voice chat stopped.")

    async def _chat_loop(self, api_key: str, secret_key: str, config_id: str):
        """Main chat loop running in async context."""
        try:
            client = AsyncHumeClient(api_key=api_key)
            options = ChatConnectOptions(config_id=config_id, secret_key=secret_key)

            async with client.empathic_voice.chat.connect_with_callbacks(
                    options=options,
                    on_open=st.session_state.handler.on_open,
                    on_message=st.session_state.handler.on_message,
                    on_close=st.session_state.handler.on_close,
                    on_error=st.session_state.handler.on_error
            ) as socket:
                st.session_state.is_connected = True

                await asyncio.create_task(
                    MicrophoneInterface.start(
                        socket,
                        allow_user_interrupt=False,
                        byte_stream=st.session_state.handler.byte_strs
                    )
                )

        except Exception as e:
            st.session_state.handler.error_message = f"Chat loop error: {str(e)}"
            st.session_state.is_connected = False

    def render_main_content(self):
        """Render the main content area."""
        st.title("🎙️ Hume Empathic Voice Chat")
        st.markdown("Real-time voice chat with emotion detection powered by Hume AI")

        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Emotions", "📝 Logs"])

        with tab1:
            self.render_chat_tab()

        with tab2:
            self.render_emotions_tab()

        with tab3:
            self.render_logs_tab()

    def render_chat_tab(self):
        """Render the chat messages tab."""
        st.subheader("Conversation")

        # Display error if any
        if hasattr(st.session_state.handler, 'error_message') and st.session_state.handler.error_message:
            st.error(st.session_state.handler.error_message)

        # Display chat messages
        if hasattr(st.session_state.handler, 'messages') and st.session_state.handler.messages:
            for msg in st.session_state.handler.messages[-10:]:  # Show last 10 messages
                timestamp = msg['timestamp'].strftime("%H:%M:%S")
                role = msg['role']
                content = msg['content']

                # Style based on role
                if role == "user":
                    st.markdown(f"**🗣️ You** ({timestamp})")
                    st.markdown(f"*{content}*")
                else:
                    st.markdown(f"**🤖 Assistant** ({timestamp})")
                    st.markdown(content)

                # Display emotions if available
                if msg.get('emotions'):
                    emotions_text = " | ".join([
                        f"{emotion}: {score:.2f}"
                        for emotion, score in msg['emotions'].items()
                    ])
                    st.caption(f"Emotions: {emotions_text}")

                st.markdown("---")
        else:
            st.info("Start a voice chat to see the conversation here.")

    def render_emotions_tab(self):
        """Render the emotions visualization tab."""
        st.subheader("Emotion Analysis")

        if hasattr(st.session_state.handler, 'emotion_scores') and st.session_state.handler.emotion_scores:
            # Current emotions bar chart
            emotions = list(st.session_state.handler.emotion_scores.keys())
            scores = list(st.session_state.handler.emotion_scores.values())

            fig = px.bar(
                x=emotions,
                y=scores,
                title="Current Emotion Scores",
                labels={'x': 'Emotions', 'y': 'Score'},
                color=scores,
                color_continuous_scale='viridis'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Emotion scores over time
            if hasattr(st.session_state.handler, 'messages') and st.session_state.handler.messages:
                emotion_history = []
                for msg in st.session_state.handler.messages:
                    if msg.get('emotions'):
                        for emotion, score in msg['emotions'].items():
                            emotion_history.append({
                                'timestamp': msg['timestamp'],
                                'emotion': emotion,
                                'score': score,
                                'role': msg['role']
                            })

                if emotion_history:
                    import pandas as pd
                    df = pd.DataFrame(emotion_history)

                    # Line chart for emotion trends
                    fig_line = px.line(
                        df,
                        x='timestamp',
                        y='score',
                        color='emotion',
                        title='Emotion Trends Over Time',
                        labels={'timestamp': 'Time', 'score': 'Emotion Score'}
                    )
                    st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Start a voice chat to see emotion analysis here.")

    def render_logs_tab(self):
        """Render the system logs tab."""
        st.subheader("System Logs")

        if hasattr(st.session_state.handler, 'logs') and st.session_state.handler.logs:
            # Display logs in reverse chronological order
            for log in reversed(st.session_state.handler.logs[-50:]):  # Show last 50 logs
                timestamp = log['timestamp']
                message = log['message']
                log_type = log['type']

                # Style based on log type
                if log_type == "error":
                    st.error(f"[{timestamp}] {message}")
                elif log_type == "system":
                    st.info(f"[{timestamp}] {message}")
                elif log_type == "metadata":
                    st.success(f"[{timestamp}] {message}")
                else:
                    st.text(f"[{timestamp}] {message}")
        else:
            st.info("System logs will appear here when you start the chat.")

    def run(self):
        """Main app entry point."""
        self.initialize_session_state()
        self.render_sidebar()
        self.render_main_content()

        # Auto-refresh every 2 seconds when chat is active
        if st.session_state.chat_started:
            time.sleep(2)
            st.rerun()


# Run the app
if __name__ == "__main__":
    app = HumeVoiceChatApp()
    app.run()