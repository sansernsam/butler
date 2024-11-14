import asyncio
from typing import Annotated
from datetime import datetime

from livekit import agents, rtc
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatImage,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero


class AssistantFunction(agents.llm.FunctionContext):
    """Enhanced assistant with visual analysis capabilities."""

    def __init__(self):
        self.latest_image = None
        self.last_analysis_time = None

    @agents.llm.ai_callable(
        description=(
            "Called when asked to evaluate something that would require vision capabilities,"
            "for example, an image, video, or the webcam feed."
        )
    )
    async def image(
        self,
        user_msg: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The user message that triggered this function"
            ),
        ],
    ):
        print(f"Message triggering vision capabilities: {user_msg}")
        self.last_analysis_time = datetime.now()
        return None


async def get_video_track(room: rtc.Room):
    """Get the first video track from the room. We'll use this track to process images."""

    video_track = asyncio.Future[rtc.RemoteVideoTrack]()

    for _, participant in room.remote_participants.items():
        for _, track_publication in participant.track_publications.items():
            if track_publication.track is not None and isinstance(
                track_publication.track, rtc.RemoteVideoTrack
            ):
                video_track.set_result(track_publication.track)
                print(f"Using video track {track_publication.track.sid}")
                break

    return await video_track


async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Room name: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "Your name is Alloy. You are a professional and friendly bilingual assistant who speaks both English "
                    "and Thai (ภาษาไทย) fluently. Your interface with users will be through voice, vision, and text. "
                    "Respond in the same language as the user's query. For Thai, use polite particles (ครับ/ค่ะ) appropriately. "
                    "You can analyze visual content when asked about what you see. "
                    "Keep responses concise and natural. Avoid using unpronounceable punctuation or emojis."
                ),
            )
        ]
    )

    # Initialize GPT-4o for enhanced capabilities
    gpt = openai.LLM(model="gpt-4o")

    # Configure TTS with enhanced Thai support
    openai_tts = tts.StreamAdapter(
        tts=openai.TTS(
            voice="alloy",
            model="tts-1-hd"  # Using HD model for better Thai pronunciation
        ),
        sentence_tokenizer=tokenize.basic.SentenceTokenizer(),
    )

    latest_image: rtc.VideoFrame | None = None

    # Initialize voice assistant with bilingual support
    assistant = VoiceAssistant(
        vad=silero.VAD.load(
            min_speech_duration=0.2,
            min_silence_duration=0.5,
        ),
        stt=deepgram.STT(
            model="nova-2",
            language="th-TH"  # Primary language set to Thai
        ),
        llm=gpt,
        tts=openai_tts,
        fnc_ctx=AssistantFunction(),
        chat_ctx=chat_context,
    )

    chat = rtc.ChatManager(ctx.room)

    async def _answer(text: str, use_image: bool = False):
        """
        Process and answer user messages with optional visual context.
        Handles both Thai and English responses appropriately.
        """
        content: list[str | ChatImage] = [text]
        if use_image and latest_image:
            content.append(ChatImage(image=latest_image))

        chat_context.messages.append(ChatMessage(role="user", content=content))

        stream = gpt.chat(chat_ctx=chat_context)
        await assistant.say(stream, allow_interruptions=True)

    @chat.on("message_received")
    def on_message_received(msg: rtc.ChatMessage):
        """Handle incoming chat messages in both Thai and English."""
        if msg.message:
            print(f"Received message: {msg.message}")
            asyncio.create_task(_answer(msg.message, use_image=False))

    @assistant.on("function_calls_finished")
    def on_function_calls_finished(called_functions: list[agents.llm.CalledFunction]):
        """Process completed function calls, particularly for visual analysis."""
        if len(called_functions) == 0:
            return

        user_msg = called_functions[0].call_info.arguments.get("user_msg")
        if user_msg:
            print(f"Processing visual analysis request: {user_msg}")
            asyncio.create_task(_answer(user_msg, use_image=True))

    # Start the assistant
    assistant.start(ctx.room)

    # Bilingual greeting
    await asyncio.sleep(1)
    await assistant.say(
        "สวัสดีครับ/ค่ะ ผม/ดิฉันคือ Alloy ยินดีให้บริการครับ/ค่ะ\nHello! I'm Alloy. How may I assist you today?",
        allow_interruptions=True
    )

    # Main loop for video processing
    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        try:
            video_track = await get_video_track(ctx.room)
            async for event in rtc.VideoStream(video_track):
                latest_image = event.frame
                await asyncio.sleep(0.1)  # Prevent high CPU usage
        except Exception as e:
            print(f"Video processing error: {e}")
            await asyncio.sleep(1)


if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
    except Exception as e:
        print(f"Application error: {e}")