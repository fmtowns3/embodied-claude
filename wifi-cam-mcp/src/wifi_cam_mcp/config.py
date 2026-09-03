"""Configuration for WiFi Camera MCP Server."""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _environment_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {value!r}")


def _default_capture_dir() -> str:
    return str(Path(tempfile.gettempdir()) / "wifi-cam-mcp")


@dataclass(frozen=True)
class CameraConfig:
    """Camera connection configuration."""

    host: str
    username: str
    password: str
    onvif_port: int = 2020
    stream_url: str | None = None
    max_width: int = 1920
    max_height: int = 1080
    mount_mode: str = "normal"  # "normal" (desktop) or "ceiling" (inverted)
    ptz_mode: str = "auto"  # "auto", "relative", or "continuous"

    @classmethod
    def from_env(cls, prefix: str = "TAPO") -> "CameraConfig":
        """Create config from environment variables.

        Args:
            prefix: Environment variable prefix (default: "TAPO")
                    For right camera, use "TAPO_RIGHT"
        """
        host = os.getenv(f"{prefix}_CAMERA_HOST", "") or os.getenv("TAPO_CAMERA_HOST", "")
        username = os.getenv(f"{prefix}_USERNAME", "") or os.getenv("TAPO_USERNAME", "")
        password = os.getenv(f"{prefix}_PASSWORD", "") or os.getenv("TAPO_PASSWORD", "")
        onvif_port = int(
            os.getenv(f"{prefix}_ONVIF_PORT", "") or os.getenv("TAPO_ONVIF_PORT", "") or "2020"
        )
        stream_url = os.getenv(f"{prefix}_STREAM_URL") or os.getenv("TAPO_STREAM_URL")
        mount_mode = (
            os.getenv(f"{prefix}_MOUNT_MODE", "") or os.getenv("TAPO_MOUNT_MODE", "") or "normal"
        ).lower()
        if mount_mode not in ("normal", "ceiling"):
            raise ValueError(f"Invalid mount mode '{mount_mode}'. Must be 'normal' or 'ceiling'.")
        ptz_mode = (
            os.getenv(f"{prefix}_PTZ_MODE", "") or os.getenv("TAPO_PTZ_MODE", "") or "auto"
        ).lower()
        if ptz_mode not in ("auto", "relative", "continuous"):
            raise ValueError(
                f"Invalid PTZ mode '{ptz_mode}'. Must be 'auto', 'relative', or 'continuous'."
            )
        max_width = int(os.getenv("CAPTURE_MAX_WIDTH", "1920"))
        max_height = int(os.getenv("CAPTURE_MAX_HEIGHT", "1080"))

        if not host:
            raise ValueError(f"{prefix}_CAMERA_HOST environment variable is required")
        if not username:
            raise ValueError(f"{prefix}_USERNAME environment variable is required")
        if not password:
            raise ValueError(f"{prefix}_PASSWORD environment variable is required")

        return cls(
            host=host,
            username=username,
            password=password,
            onvif_port=onvif_port,
            stream_url=stream_url,
            mount_mode=mount_mode,
            ptz_mode=ptz_mode,
            max_width=max_width,
            max_height=max_height,
        )

    @classmethod
    def right_camera_from_env(cls) -> "CameraConfig | None":
        """Create config for right camera if configured.

        Returns:
            CameraConfig for right camera, or None if not configured
        """
        host = os.getenv("TAPO_RIGHT_CAMERA_HOST", "")
        if not host:
            return None

        # Right camera can share username/password with left, or have its own
        username = os.getenv("TAPO_RIGHT_USERNAME", "") or os.getenv("TAPO_USERNAME", "")
        password = os.getenv("TAPO_RIGHT_PASSWORD", "") or os.getenv("TAPO_PASSWORD", "")
        onvif_port = int(
            os.getenv("TAPO_RIGHT_ONVIF_PORT", "") or os.getenv("TAPO_ONVIF_PORT", "") or "2020"
        )
        stream_url = os.getenv("TAPO_RIGHT_STREAM_URL")
        mount_mode = (
            os.getenv("TAPO_RIGHT_MOUNT_MODE", "") or os.getenv("TAPO_MOUNT_MODE", "") or "normal"
        ).lower()
        if mount_mode not in ("normal", "ceiling"):
            raise ValueError(f"Invalid mount mode '{mount_mode}'. Must be 'normal' or 'ceiling'.")
        ptz_mode = (
            os.getenv("TAPO_RIGHT_PTZ_MODE", "") or os.getenv("TAPO_PTZ_MODE", "") or "auto"
        ).lower()
        if ptz_mode not in ("auto", "relative", "continuous"):
            raise ValueError(
                f"Invalid PTZ mode '{ptz_mode}'. Must be 'auto', 'relative', or 'continuous'."
            )
        max_width = int(os.getenv("CAPTURE_MAX_WIDTH", "1920"))
        max_height = int(os.getenv("CAPTURE_MAX_HEIGHT", "1080"))

        if not username or not password:
            return None

        return cls(
            host=host,
            username=username,
            password=password,
            onvif_port=onvif_port,
            stream_url=stream_url,
            mount_mode=mount_mode,
            ptz_mode=ptz_mode,
            max_width=max_width,
            max_height=max_height,
        )


@dataclass(frozen=True)
class ServerConfig:
    """MCP Server configuration."""

    name: str = "wifi-cam-mcp"
    version: str = "0.4.6"
    capture_dir: str = field(default_factory=_default_capture_dir)
    mic_source: str = "camera"  # "camera" (RTSP) or "local" (PC microphone)
    mic_device: str | None = None  # DirectShow device name for Windows local mic
    transcribe_default: bool = True
    # "openai-whisper" / "faster-whisper" (both local), or "openai-api" (cloud)
    transcribe_backend: str = "openai-whisper"
    # Whisper model size (local) or OpenAI model id (openai-api)
    transcribe_model: str = "base"
    openai_api_key: str | None = None  # required when transcribe_backend == "openai-api"

    @classmethod
    def from_env(cls) -> "ServerConfig":
        """Create config from environment variables."""
        mic_source = os.getenv("MIC_SOURCE", "camera").lower()
        if mic_source not in ("camera", "local"):
            raise ValueError(f"Invalid MIC_SOURCE '{mic_source}'. Must be 'camera' or 'local'.")
        transcribe_backend = os.getenv("TRANSCRIBE_BACKEND", "openai-whisper").lower()
        if transcribe_backend not in ("openai-whisper", "faster-whisper", "openai-api"):
            raise ValueError(
                f"Invalid TRANSCRIBE_BACKEND '{transcribe_backend}'. Must be "
                "'openai-whisper', 'faster-whisper', or 'openai-api'."
            )
        capture_dir = os.getenv("CAPTURE_DIR", "").strip() or _default_capture_dir()
        # TRANSCRIBE_MODEL is a Whisper size for the local backends, or an
        # OpenAI model id for openai-api, so the default differs per backend.
        #
        # The openai-api default is whisper-1, not gpt-4o-transcribe, because
        # MIC_SOURCE defaults to the camera and a Tapo RTSP audio track is
        # pcm_alaw 8000 Hz. Measured on that band, gpt-4o-transcribe alters the
        # first mora of a Japanese proper noun (a voiced bilabial stop comes
        # back as a nasal) while whisper-1 keeps it. One mora is not a slightly
        # worse transcript here: the sociality layer keys on proper nouns, so a
        # name that arrives one mora off is a different person. Nothing reports
        # it either; the transcript comes back looking like a transcript.
        #
        # On wideband audio (MIC_SOURCE=local, a PC microphone) gpt-4o-transcribe
        # is both more accurate and much faster; set TRANSCRIBE_MODEL explicitly
        # in that case.
        default_model = "whisper-1" if transcribe_backend == "openai-api" else "base"
        return cls(
            name=os.getenv("MCP_SERVER_NAME", "wifi-cam-mcp"),
            version=os.getenv("MCP_SERVER_VERSION", "0.4.6"),
            capture_dir=capture_dir,
            mic_source=mic_source,
            mic_device=os.getenv("MIC_DEVICE") or None,
            transcribe_default=_environment_bool("TRANSCRIBE_DEFAULT", True),
            transcribe_backend=transcribe_backend,
            transcribe_model=os.getenv("TRANSCRIBE_MODEL", default_model),
            openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        )
