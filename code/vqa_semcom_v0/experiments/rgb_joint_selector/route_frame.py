"""One-byte visual-tier prefix around an unchanged complete neural image packet."""
TIERS = ("low", "medium", "high")


def pack(payload: bytes, tier: str) -> bytes:
    if not isinstance(payload, bytes) or not payload or tier not in TIERS:
        raise ValueError("Nonempty codec bytes and a registered visual tier are required")
    return bytes([TIERS.index(tier)]) + payload


def unpack(frame: bytes) -> tuple[str, bytes]:
    if not isinstance(frame, bytes) or len(frame) < 2 or frame[0] >= len(TIERS):
        raise ValueError("Invalid or empty neural-image routing frame")
    return TIERS[frame[0]], frame[1:]
