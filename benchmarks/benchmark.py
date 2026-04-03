from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from src.ccllm.compression import TextCompressor, TextDecompressor


@dataclass(frozen=True)
class BenchmarkResult:
    sample_name: str
    method: str
    runs: int
    original_chars: int
    original_bytes: int
    original_token_count: int
    compressed_bytes: int
    compressed_token_count: int | None
    compression_ratio: float
    space_saving_ratio: float
    avg_compress_ms: float
    avg_decompress_ms: float
    round_trip_ok: bool
    package_json_bytes: int


class CompressionBenchmark:
    """
    Benchmark the compression subsystem across supported methods.

    Measures:
    - compression time
    - decompression time
    - compressed payload size
    - package JSON size
    - token statistics
    - round-trip correctness

    Supported methods:
    - none
    - zlib
    - lzma
    - dictionary
    """

    def __init__(self, runs: int = 5) -> None:
        if runs < 1:
            raise ValueError("runs must be >= 1")
        self.runs = runs

    def benchmark_text(self, sample_name: str, text: str) -> list[BenchmarkResult]:
        methods = ["none", "zlib", "lzma", "dictionary"]
        results: list[BenchmarkResult] = []

        for method in methods:
            compressor = self._build_compressor(method)
            decompressor = TextDecompressor()

            compress_times_ms: list[float] = []
            decompress_times_ms: list[float] = []

            package = None
            restored_text = None

            for _ in range(self.runs):
                start = time.perf_counter()
                package = compressor.compress(
                    text, metadata={"sample_name": sample_name}
                )
                end = time.perf_counter()
                compress_times_ms.append((end - start) * 1000.0)

                start = time.perf_counter()
                restored_text = decompressor.decompress(package)
                end = time.perf_counter()
                decompress_times_ms.append((end - start) * 1000.0)

            if package is None or restored_text is None:
                raise RuntimeError("Benchmark failed to produce a compression package")

            result = BenchmarkResult(
                sample_name=sample_name,
                method=method,
                runs=self.runs,
                original_chars=len(text),
                original_bytes=len(text.encode("utf-8")),
                original_token_count=package.token_count,
                compressed_bytes=package.stats.compressed_bytes,
                compressed_token_count=package.stats.compressed_token_count,
                compression_ratio=package.stats.compression_ratio,
                space_saving_ratio=package.stats.space_saving_ratio,
                avg_compress_ms=statistics.mean(compress_times_ms),
                avg_decompress_ms=statistics.mean(decompress_times_ms),
                round_trip_ok=(restored_text == text),
                package_json_bytes=len(package.to_json().encode("utf-8")),
            )
            results.append(result)

        return results

    @staticmethod
    def _build_compressor(method: str) -> TextCompressor:
        if method == "dictionary":
            return TextCompressor(
                method="dictionary",
                min_phrase_len=2,
                max_phrase_len=8,
                min_frequency=2,
                max_dictionary_size=256,
                min_estimated_savings=1,
                skip_all_whitespace_phrases=True,
            )
        return TextCompressor(method=method)

    @staticmethod
    def default_samples() -> dict[str, str]:
        return {
            "short_prompt": (
                "You are a helpful assistant.\n"
                "Summarize the following paragraph clearly and accurately."
            ),
            "repeated_prompt": (
                "You are a helpful assistant. " * 40
                + "Answer carefully. "
                + "Do not omit important details. " * 20
            ).strip(),
            "structured_markdown": (
                "# Task\n"
                "- Summarize the document\n"
                "- Preserve all numbers\n"
                "- Keep section names unchanged\n\n"
                "## Constraints\n"
                "1. No hallucination\n"
                "2. No omission\n"
                "3. Maintain factual consistency\n\n"
            )
            * 10,
            "json_like_text": (
                '{"role":"system","content":"You are a helpful assistant."}\n'
                '{"role":"user","content":"Extract the key entities and keep '
                'the values exact."}\n'
            )
            * 20,
            "whitespace_heavy": (
                "Prompt:\n"
                "    - item one\n"
                "    - item two\n"
                "    - item three\n\n"
                "Response:\t\tKeep formatting exact.\n"
            )
            * 20,
            "long_context": (
                "The compression system must support exact reconstruction. "
                "The compression system must support exact reconstruction. "
                "The benchmark should compare methods fairly. "
                "The benchmark should compare methods fairly. "
                "Token-aware phrase substitution is useful for "
                "repeated prompt patterns. "
                "Token-aware phrase substitution is useful for "
                "repeated prompt patterns. "
            )
            * 80,
        }

    @staticmethod
    def format_results_table(results: list[BenchmarkResult]) -> str:
        headers = [
            "sample",
            "method",
            "orig_bytes",
            "comp_bytes",
            "comp_ratio",
            "save_ratio",
            "comp_ms",
            "decomp_ms",
            "tok_orig",
            "tok_comp",
            "ok",
        ]

        rows: list[list[str]] = []
        for result in results:
            rows.append(
                [
                    result.sample_name,
                    result.method,
                    str(result.original_bytes),
                    str(result.compressed_bytes),
                    f"{result.compression_ratio:.4f}",
                    f"{result.space_saving_ratio:.4f}",
                    f"{result.avg_compress_ms:.3f}",
                    f"{result.avg_decompress_ms:.3f}",
                    str(result.original_token_count),
                    (
                        str(result.compressed_token_count)
                        if result.compressed_token_count is not None
                        else "-"
                    ),
                    "yes" if result.round_trip_ok else "no",
                ]
            )

        widths = [
            max(len(headers[idx]), max(len(row[idx]) for row in rows))
            for idx in range(len(headers))
        ]

        def fmt_row(row: list[str]) -> str:
            return " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))

        separator = "-+-".join("-" * width for width in widths)
        lines = [fmt_row(headers), separator]
        lines.extend(fmt_row(row) for row in rows)
        return "\n".join(lines)


def save_results_json(
    results: list[BenchmarkResult],
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(result) for result in results]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def collect_samples_from_directory(input_dir: str | Path) -> dict[str, str]:
    path = Path(input_dir)
    if not path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Input path is not a directory: {path}")

    samples: dict[str, str] = {}
    for file_path in sorted(path.glob("*.txt")):
        samples[file_path.stem] = file_path.read_text(encoding="utf-8")

    if not samples:
        raise ValueError(f"No .txt files found in directory: {path}")

    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark text compression methods for the compression module."
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of repeated timing runs per sample per method.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Optional directory containing .txt files to benchmark.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="benchmarks/results/benchmark_results.json",
        help="Path to save JSON benchmark results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = CompressionBenchmark(runs=args.runs)

    if args.input_dir:
        samples = collect_samples_from_directory(args.input_dir)
    else:
        samples = benchmark.default_samples()

    all_results: list[BenchmarkResult] = []

    for sample_name, text in samples.items():
        results = benchmark.benchmark_text(sample_name=sample_name, text=text)
        all_results.extend(results)

    print(benchmark.format_results_table(all_results))
    save_results_json(all_results, args.output_json)
    print(f"\nSaved benchmark results to: {args.output_json}")


if __name__ == "__main__":
    main()
