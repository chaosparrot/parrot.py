"""How the GUI words times and durations. No Qt, so it tests like lib/."""

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def duration_text(seconds):
    """'7.4s', '1m 04s', '1h 03m'."""
    if round(seconds, 1) < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def spoken_duration(seconds):
    """'1 minute 4 seconds', for a screen reader. '1m 04s' can be read as
    metres, and '01:04' digit by digit."""
    if 0 < seconds < 0.5:
        return "under a second"
    # Half up: round() goes to even, so 0.5 would read "0 seconds"
    hours, rest = divmod(int(seconds + 0.5), 3600)
    minutes, secs = divmod(rest, 60)
    parts = [f"{n} {unit}{'' if n == 1 else 's'}"
             for n, unit in ((hours, "hour"), (minutes, "minute"), (secs, "second")) if n]
    return " ".join(parts) or "0 seconds"


def recorded_at(stamp):
    """'Sep 02, 2023, 10:20 AM'. The time matters: several takes of one sound
    in a sitting is the normal way to record, and a date alone leaves them
    reading identically. Built by hand: strftime's month and AM/PM follow the
    system locale, and some locales have no AM/PM at all."""
    hour = stamp.hour % 12 or 12
    half = "AM" if stamp.hour < 12 else "PM"
    return (f"{_MONTHS[stamp.month - 1]} {stamp.day:02d}, {stamp.year}, "
            f"{hour}:{stamp.minute:02d} {half}")
