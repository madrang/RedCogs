# The content-refusal checks of the image endpoints: the moderation headers,
# the pixel blank check, and the refusal verdict built from both.

import io

from .catalog import VENICE_IMAGE_DARK_MAX

# The pixel decoder of the blank-refusal check. Guarded: without Pillow
# the check cannot run and a flagged answer posts with its [venice] line.
try:
    from PIL import Image, ImageStat
except ImportError:
    Image = None


def _header_flag(headers, name: str) -> str:
    """The yes/no/unknown text of a boolean response header."""
    value = headers.get(name) if headers is not None else None
    if value is None:
        return "unknown"
    return "yes" if str(value).strip().lower() == "true" else "no"


def _moderation_status(headers) -> tuple[str, str]:
    """The (violation flag, [venice] status line) of an image answer.
       A flag reaches the model only when it reads yes: a clean answer carries no line at all.
       The generate and the edit endpoint answer the same headers.
    """
    violation = _header_flag(headers, "x-venice-is-content-violation")
    raised = [
        f"{label}: {flag}"
        for label, flag in (
            ("content violation", violation)
            , ("blurred", _header_flag(headers, "x-venice-is-blurred"))
        )
        if flag == "yes"
    ]
    return violation, f"[venice] {', '.join(raised)}" if raised else ""


def _blank_check(data: bytes) -> bool | None:
    """True when the image is a uniform fill or near-black, False when it holds content, None when no decoder can read it."""
    if Image is None:
        return None
    try:
        with Image.open(io.BytesIO(data)) as decoded:
            picture = decoded.convert("RGB")
            colors = picture.getcolors(maxcolors=2)
            if colors is not None:
                # Two colors at most: one is a uniform fill, two is content.
                return len(colors) == 1
            stat = ImageStat.Stat(picture)
            return all(maximum <= VENICE_IMAGE_DARK_MAX for _minimum, maximum in stat.extrema)
    except Exception:
        return None


def _refusal_error(what: str, data: bytes, violation: str, status: str) -> str | None:
    """The error text when the endpoint refused the request, else None: the image posts.
       A refusal needs the violation flag and the pixel check reading blank.
       No size stand-in: without a decoder the check cannot run and a flagged answer posts (the [venice] line already names the flag),
       a real render posts the same way.
    """
    if violation != "yes":
        return None
    if _blank_check(data) is not True:
        return None
    return (
        f"Error: the {what} was refused: the answer is a blank image of {len(data)} bytes "
        f"and the endpoint flags a content violation. {status}. Nothing was posted. "
        "Describe the subject instead of naming it, or pick a model without the refusal flag."
    )
