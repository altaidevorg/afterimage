"""Tests for SVG chart generators."""

from afterimage.analytics.charts import bar_chart, donut_chart, histogram, metric_card


class TestMetricCard:
    def test_renders_svg(self):
        svg = metric_card("Total", "42")
        assert "<svg" in svg
        assert "42" in svg
        assert "Total" in svg

    def test_subtitle(self):
        svg = metric_card("Entropy", "3.2", subtitle="bits")
        assert "bits" in svg

    def test_escapes_html(self):
        svg = metric_card("Test <b>", "1&2")
        assert "&lt;b&gt;" in svg
        assert "1&amp;2" in svg


class TestBarChart:
    def test_empty(self):
        assert bar_chart({}) == ""

    def test_basic(self):
        svg = bar_chart({"alpha": 10, "beta": 5})
        assert "<svg" in svg
        assert "alpha" in svg
        assert "beta" in svg

    def test_with_title(self):
        svg = bar_chart({"a": 1}, title="My Chart")
        assert "My Chart" in svg

    def test_max_bars_limits(self):
        data = {f"item{i}": i for i in range(30)}
        svg = bar_chart(data, max_bars=5)
        # Should only contain 5 bars
        assert svg.count("<rect") <= 6  # 5 bars + possible title bg

    def test_long_labels_truncated(self):
        svg = bar_chart({"a" * 30: 10})
        assert "..." in svg


class TestHistogram:
    def test_empty(self):
        assert histogram([], []) == ""

    def test_basic(self):
        svg = histogram([3, 5, 2, 1], ["0", "10", "20", "30"])
        assert "<svg" in svg
        assert "5" in svg  # max count visible

    def test_with_title(self):
        svg = histogram([1, 2], ["a", "b"], title="Lengths")
        assert "Lengths" in svg


class TestDonutChart:
    def test_empty(self):
        assert donut_chart({}) == ""

    def test_zero_total(self):
        assert donut_chart({"a": 0, "b": 0}) == ""

    def test_basic(self):
        svg = donut_chart({"good": 10, "bad": 3})
        assert "<svg" in svg
        assert "good" in svg
        assert "bad" in svg
        assert "13" in svg  # total in center

    def test_with_title(self):
        svg = donut_chart({"a": 1}, title="Grades")
        assert "Grades" in svg

    def test_single_segment(self):
        svg = donut_chart({"only": 5})
        assert "<svg" in svg
        assert "5" in svg
