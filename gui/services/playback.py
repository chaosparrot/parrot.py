import threading
import time

import numpy as np
import sounddevice as sd

_SAMPLES_PER_CHUNK = 256


class PlaybackError(Exception):
    pass


class _Player:
    def __init__(self):
        self._lock = threading.Lock()
        self._current_play_id = None
        self._playback_thread = None
        self._failed_mid_play = False

    def play(self, samples, sample_rate):
        self.stop()
        self._wait_for_playback_thread()
        if samples is None or not sample_rate or len(samples) == 0:
            return 0.0
        try:
            stream = sd.OutputStream(samplerate=sample_rate, channels=1,
                                     dtype="float32", blocksize=0,
                                     latency="low")
        except Exception as exc:
            raise PlaybackError(str(exc)) from exc
        try:
            stream.start()
        except Exception as exc:
            stream.close(ignore_errors=True)
            raise PlaybackError(str(exc)) from exc
        play_id = object()
        playback_thread = threading.Thread(
            target=self._playback_loop, args=(samples, stream, play_id), daemon=True)
        with self._lock:
            self._current_play_id = play_id
            self._playback_thread = playback_thread
            self._failed_mid_play = False
        playback_thread.start()
        return float(stream.latency)

    def stop(self):
        with self._lock:
            self._current_play_id = None

    def is_playing(self):
        return self._current_play_id is not None

    def failed_mid_play(self):
        return self._failed_mid_play

    def shutdown(self):
        self.stop()
        self._wait_for_playback_thread()

    def _wait_for_playback_thread(self):
        with self._lock:
            playback_thread = self._playback_thread
        if playback_thread is not None:
            playback_thread.join(timeout=2.0)

    def _playback_loop(self, samples, stream, play_id):
        try:
            for i in range(0, len(samples), _SAMPLES_PER_CHUNK):
                if self._current_play_id is not play_id:
                    break
                chunk = np.ascontiguousarray(samples[i:i + _SAMPLES_PER_CHUNK],
                                             dtype=np.float32)
                stream.write(chunk.reshape(-1, 1))
            else:  # for loop finished without break - played to end
                self._wait_for_last_chunk_to_play(stream, play_id)
        except sd.PortAudioError:
            with self._lock:
                if self._current_play_id is play_id:
                    self._failed_mid_play = True     # e.g. headset unplugged
        finally:
            with self._lock:
                if self._current_play_id is play_id:
                    self._current_play_id = None
            stream.abort(ignore_errors=True)
            stream.close(ignore_errors=True)

    def _wait_for_last_chunk_to_play(self, stream, play_id):
        heard_by = time.monotonic() + stream.latency + 0.02
        while self._current_play_id is play_id and time.monotonic() < heard_by:
            time.sleep(0.01)


_player = _Player()

play = _player.play
stop = _player.stop
is_playing = _player.is_playing
failed_mid_play = _player.failed_mid_play
shutdown = _player.shutdown
