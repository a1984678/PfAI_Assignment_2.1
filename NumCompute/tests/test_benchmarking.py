import pytest

from numcompute import benchmarking


class TestTimeIt:

    def test_returns_positive_float(self):
        t = benchmarking.time_it(sum, range(1000))
        assert isinstance(t, float) and t >= 0

    def test_call_count_is_warmup_plus_repeats(self):
        calls = []
        benchmarking.time_it(lambda: calls.append(1), repeats=4, warmup=2)
        assert len(calls) == 6

    def test_no_warmup(self):
        calls = []
        benchmarking.time_it(lambda: calls.append(1), repeats=3, warmup=0)
        assert len(calls) == 3

    def test_passes_args_and_kwargs(self):
        seen = []
        benchmarking.time_it(lambda a, b=0: seen.append((a, b)), 1, b=2, repeats=1, warmup=0)
        assert seen == [(1, 2)]

    def test_not_callable_raises(self):
        with pytest.raises(ValueError, match="callable"):
            benchmarking.time_it(5)

    @pytest.mark.parametrize("repeats", [0, -1, 1.5])
    def test_bad_repeats_raises(self, repeats):
        with pytest.raises(ValueError, match="repeats"):
            benchmarking.time_it(sum, [1], repeats=repeats)

    def test_bad_warmup_raises(self):
        with pytest.raises(ValueError, match="warmup"):
            benchmarking.time_it(sum, [1], warmup=-1)


class TestCompare:

    def test_labels_and_order_kept(self):
        out = benchmarking.compare({"a": sum, "b": max}, [1, 2, 3], repeats=2)
        assert list(out) == ["a", "b"]
        assert all(v >= 0 for v in out.values())

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty dict"):
            benchmarking.compare({})

    def test_not_a_dict_raises(self):
        with pytest.raises(ValueError, match="non-empty dict"):
            benchmarking.compare([sum])


class TestReport:

    def test_baseline_defaults_to_first(self):
        text = benchmarking.report({"loop": 0.010, "numpy": 0.002})
        lines = text.splitlines()
        assert "(baseline)" in lines[0] and "5.0x faster" in lines[1]

    def test_slower_candidate(self):
        text = benchmarking.report({"fast": 0.001, "slow": 0.004})
        assert "4.0x slower" in text.splitlines()[1]

    def test_named_baseline(self):
        text = benchmarking.report({"a": 0.002, "b": 0.001}, baseline="b")
        assert "(baseline)" in text.splitlines()[1]

    def test_works_with_compare_output(self):
        text = benchmarking.report(benchmarking.compare({"sum": sum}, [1, 2], repeats=1))
        assert "sum" in text and "ms" in text

    def test_unknown_baseline_raises(self):
        with pytest.raises(ValueError, match="not in results"):
            benchmarking.report({"a": 1.0}, baseline="z")

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="non-empty dict"):
            benchmarking.report({})

    def test_non_positive_time_raises(self):
        with pytest.raises(ValueError, match="must be positive"):
            benchmarking.report({"a": 0.0})


class TestEnvironmentInfo:

    def test_keys_and_types(self):
        info = benchmarking.environment_info()
        assert set(info) == {"python", "numpy", "platform", "processor"}
        assert all(isinstance(v, str) and v for v in info.values())

    def test_numpy_version_matches(self):
        import numpy as np
        assert benchmarking.environment_info()["numpy"] == np.__version__


class TestLoopVsVectorised:

    @staticmethod
    def _loop(x):
        total = 0.0
        for v in x:
            total += v * v
        return total

    @staticmethod
    def _vec(x):
        return float((x * x).sum())

    def test_returns_times_and_speedup(self):
        import numpy as np
        out = benchmarking.compare_loop_vs_vectorised(self._loop, self._vec,
                                                      np.arange(20000.0), repeats=2)
        assert set(out) == {"loop", "vectorised", "speedup"}
        assert out["loop"] > 0 and out["vectorised"] > 0

    def test_vectorised_is_faster_on_large_input(self):
        import numpy as np
        out = benchmarking.compare_loop_vs_vectorised(self._loop, self._vec,
                                                      np.arange(200000.0), repeats=3)
        assert out["speedup"] > 1

    def test_different_results_raise(self):
        with pytest.raises(ValueError, match="different results"):
            benchmarking.compare_loop_vs_vectorised(lambda: 1.0, lambda: 2.0)

    def test_different_shapes_raise(self):
        import numpy as np
        with pytest.raises(ValueError, match="different results"):
            benchmarking.compare_loop_vs_vectorised(lambda: np.zeros(2), lambda: np.zeros(3))
