"""Tests for SWA window state model."""

from megakernel_lab.swa_state import SlidingWindowState


class TestSlidingWindowState:
    def test_window_size_is_enforced(self) -> None:
        swa = SlidingWindowState(window_size=8)
        long_prefix = list(range(100))
        window = swa.get_window_tokens(long_prefix, 100)
        assert len(window) == 8
        assert window == list(range(92, 100))

    def test_short_prefix_returns_all(self) -> None:
        swa = SlidingWindowState(window_size=256)
        short = [1, 2, 3]
        window = swa.get_window_tokens(short, 3)
        assert window == [1, 2, 3]

    def test_state_read_counter_updates(self) -> None:
        swa = SlidingWindowState()
        assert swa.total_state_reads == 0
        swa.read_window_state(1, 10)
        assert swa.total_state_reads == 1
        swa.read_window_state(2, 20)
        assert swa.total_state_reads == 2

    def test_state_write_counter_updates(self) -> None:
        swa = SlidingWindowState()
        assert swa.total_state_writes == 0
        swa.update_window_state(1, [10, 20, 30])
        assert swa.total_state_writes == 1

    def test_update_after_commit_changes_state_position(self) -> None:
        swa = SlidingWindowState()
        swa.update_window_state(1, [10, 20, 30])
        assert swa.kv_or_state_pages[1] == [10, 20, 30]
        swa.update_window_state(1, [10, 20, 30, 40, 50])
        assert len(swa.kv_or_state_pages[1]) == 5

    def test_report_returns_summary(self) -> None:
        swa = SlidingWindowState(window_size=128)
        swa.read_window_state(1, 0)
        swa.update_window_state(1, [10, 20])
        report = swa.report()
        assert report["window_size"] == 128
        assert report["total_state_reads"] == 1
        assert report["total_state_writes"] == 1
        assert report["active_requests"] == 1
