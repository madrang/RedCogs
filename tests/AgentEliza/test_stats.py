"""The media windows of the credit ratio: the bands and the lerped slide."""

from AgentEliza.stats import MEDIA_RATE_CHANNEL, MEDIA_RATE_USER, media_windows


def test_the_full_windows_hold_at_and_above_the_top_band() -> None:
    full = {"user": MEDIA_RATE_USER, "channel": MEDIA_RATE_CHANNEL}
    # An unread ratio never blocks: the full windows hold.
    assert media_windows(None) == full
    assert media_windows(1.25) == full
    assert media_windows(1.9) == full


def test_the_windows_slide_linearly_to_one_at_the_floor_band() -> None:
    # Two checkpoints of the slide: a fifth and four fifths of the band.
    assert media_windows(0.97) == {"user": 2, "channel": 7}
    assert media_windows(1.18) == {"user": 7, "channel": 26}
    # The floor of the band reads one generation on both windows.
    assert media_windows(0.9) == {"user": 1, "channel": 1}
    assert media_windows(0.91) == {"user": 1, "channel": 2}


def test_one_generation_per_hour_holds_above_the_disable_band() -> None:
    assert media_windows(0.89) == {"user": 1, "channel": 1}
    assert media_windows(0.75) == {"user": 1, "channel": 1}


def test_the_disable_band_closes_the_media_tools() -> None:
    assert media_windows(0.74) is None
    assert media_windows(0.0) is None
