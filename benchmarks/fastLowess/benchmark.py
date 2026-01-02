"""
Industry-level LOWESS benchmarks with JSON output for comparison with Statsmodels.

Benchmarks are aligned with the Rust criterion benchmarks and Statsmodels benchmarks.
Results are written to benchmarks/output/fastlowess_benchmark.json.

Run with: python3 benchmark.py
"""

import json
import time
import numpy as np
import fastlowess
from pathlib import Path
from typing import Tuple, List, Dict
from dataclasses import dataclass, asdict

# ============================================================================
# Benchmark Result Storage
# ============================================================================

@dataclass
class BenchmarkResult:
    """Store benchmark timing results."""
    name: str
    size: int
    iterations: int
    mean_time_ms: float = 0.0
    std_time_ms: float = 0.0
    median_time_ms: float = 0.0
    min_time_ms: float = 0.0
    max_time_ms: float = 0.0


def run_benchmark(name: str, size: int, func, iterations: int = 10, warmup: int = 2) -> BenchmarkResult:
    """Run a benchmark with warmup and timing."""
    result = BenchmarkResult(name=name, size=size, iterations=iterations)
    
    # Warmup runs
    for _ in range(warmup):
        func()
    
    # Timed runs
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms
    
    result.mean_time_ms = np.mean(times)
    result.std_time_ms = np.std(times)
    result.median_time_ms = np.median(times)
    result.min_time_ms = np.min(times)
    result.max_time_ms = np.max(times)
    
    return result


# ============================================================================
# Data Generation (Aligned with Rust)
# ============================================================================

def generate_sine_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10.0, size)
    y = np.sin(x) + rng.normal(0.0, 0.2, size)
    return x, y

def generate_outlier_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10.0, size)
    y = np.sin(x) + rng.normal(0.0, 0.2, size)
    
    n_outliers = size // 20
    outlier_indices = rng.integers(0, size, n_outliers)
    y[outlier_indices] += rng.uniform(-5.0, 5.0, n_outliers)
    
    return x, y

def generate_financial_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.arange(size, dtype=float)
    y = np.empty(size, dtype=float)
    y[0] = 100.0
    returns = rng.normal(0.0005, 0.02, size - 1)
    y[1:] = 100.0 * np.cumprod(1.0 + returns)
    return x, y

def generate_scientific_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, size * 0.01, size)
    signal = np.exp(-x * 0.3) * np.cos(x * 2.0 * np.pi)
    y = signal + rng.normal(0.0, 0.05, size)
    return x, y

def generate_genomic_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.arange(size, dtype=float) * 1000.0
    base = 0.5 + np.sin(x / 50000.0) * 0.3
    noise = rng.normal(0.0, 0.1, size)
    y = np.clip(base + noise, 0.0, 1.0)
    return x, y

def generate_clustered_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(size)
    x = (indices // 100).astype(float) + (indices % 100).astype(float) * 1e-6
    y = np.sin(x) + rng.normal(0.0, 0.1, size)
    return x, y

def generate_high_noise_data(size: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10.0, size)
    y = np.sin(x) * 0.5 + rng.normal(0.0, 2.0, size)
    return x, y

# ============================================================================
# Benchmark Categories
# ============================================================================

def benchmark_scalability(iterations: int = 10, parallel: bool = True) -> List[BenchmarkResult]:
    """Benchmark performance scaling with dataset size."""
    print("\n" + "=" * 80)
    print(f"SCALABILITY (Parallel={parallel})")
    print("=" * 80)
    
    results = []
    sizes = [1000, 5000, 10000, 50000, 100000]
    
    for size in sizes:
        x, y = generate_sine_data(size, seed=42)
        
        def run():
            fastlowess.smooth(
                x, y,
                fraction=0.1,
                iterations=3,
                scaling_method="mar",
                boundary_policy="noboundary",
                parallel=parallel
            )
        
        result = run_benchmark(f"scale_{size}", size, run, iterations)
        results.append(result)
        print(f"  scale_{size}: {result.mean_time_ms:.4f} ms ± {result.std_time_ms:.4f} ms")
    
    return results


def benchmark_fraction(iterations: int = 10, parallel: bool = True) -> List[BenchmarkResult]:
    """Benchmark different smoothing fractions."""
    print("\n" + "=" * 80)
    print(f"FRACTION (Parallel={parallel})")
    print("=" * 80)
    
    results = []
    size = 5000
    fractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.67]
    x, y = generate_sine_data(size, seed=42)
    
    for frac in fractions:
        def run(f=frac):
            fastlowess.smooth(
                x, y,
                fraction=f,
                iterations=3,
                scaling_method="mar",
                boundary_policy="noboundary",
                parallel=parallel
            )
        
        result = run_benchmark(f"fraction_{frac}", size, run, iterations)
        results.append(result)
        print(f"  fraction_{frac}: {result.mean_time_ms:.4f} ms ± {result.std_time_ms:.4f} ms")
    
    return results


def benchmark_iterations(iterations: int = 10, parallel: bool = True) -> List[BenchmarkResult]:
    """Benchmark different robustness iterations."""
    print("\n" + "=" * 80)
    print(f"ITERATIONS (Parallel={parallel})")
    print("=" * 80)
    
    results = []
    size = 5000
    iter_values = [0, 1, 2, 3, 5, 10]
    x, y = generate_outlier_data(size, seed=42)
    
    for it in iter_values:
        def run(n=it):
            fastlowess.smooth(
                x, y,
                fraction=0.2,
                iterations=n,
                scaling_method="mar",
                boundary_policy="noboundary",
                parallel=parallel
            )
        
        result = run_benchmark(f"iterations_{it}", size, run, iterations)
        results.append(result)
        print(f"  iterations_{it}: {result.mean_time_ms:.4f} ms ± {result.std_time_ms:.4f} ms")
    
    return results


def benchmark_delta(iterations: int = 10, parallel: bool = True) -> List[BenchmarkResult]:
    """Benchmark delta parameter effects."""
    print("\n" + "=" * 80)
    print(f"DELTA (Parallel={parallel})")
    print("=" * 80)
    
    results = []
    size = 10000
    x, y = generate_sine_data(size, seed=42)
    
    delta_configs = [
        ("delta_none", 0.0),
        ("delta_small", 0.5),
        ("delta_medium", 2.0),
        ("delta_large", 10.0),
    ]
    
    for name, delta in delta_configs:
        def run(d=delta):
            fastlowess.smooth(
                x, y,
                fraction=0.2,
                iterations=2,
                delta=d,
                scaling_method="mar",
                boundary_policy="noboundary",
                parallel=parallel
            )
        
        result = run_benchmark(name, size, run, iterations)
        results.append(result)
        print(f"  {name}: {result.mean_time_ms:.4f} ms ± {result.std_time_ms:.4f} ms")
    
    return results


def benchmark_financial(iterations: int = 10, parallel: bool = True) -> List[BenchmarkResult]:
    """Benchmark with financial time series data."""
    print("\n" + "=" * 80)
    print(f"FINANCIAL (Parallel={parallel})")
    print("=" * 80)
    
    results = []
    sizes = [500, 1000, 5000, 10000]
    
    for size in sizes:
        x, y = generate_financial_data(size, seed=42)
        
        def run():
            fastlowess.smooth(
                x, y,
                fraction=0.1,
                iterations=2,
                scaling_method="mar",
                boundary_policy="noboundary",
                parallel=parallel
            )
        
        result = run_benchmark(f"financial_{size}", size, run, iterations)
        results.append(result)
        print(f"  financial_{size}: {result.mean_time_ms:.4f} ms ± {result.std_time_ms:.4f} ms")
    
    return results


def benchmark_scientific(iterations: int = 10, parallel: bool = True) -> List[BenchmarkResult]:
    """Benchmark with scientific measurement data."""
    print("\n" + "=" * 80)
    print(f"SCIENTIFIC (Parallel={parallel})")
    print("=" * 80)
    
    results = []
    sizes = [500, 1000, 5000, 10000]
    
    for size in sizes:
        x, y = generate_scientific_data(size, seed=42)
        
        def run():
            fastlowess.smooth(
                x, y,
                fraction=0.15,
                iterations=3,
                scaling_method="mar",
                boundary_policy="noboundary",
                parallel=parallel
            )
        
        result = run_benchmark(f"scientific_{size}", size, run, iterations)
        results.append(result)
        print(f"  scientific_{size}: {result.mean_time_ms:.4f} ms ± {result.std_time_ms:.4f} ms")
    
    return results


def benchmark_genomic(iterations: int = 10, parallel: bool = True) -> List[BenchmarkResult]:
    """Benchmark with genomic methylation data."""
    print("\n" + "=" * 80)
    print(f"GENOMIC (Parallel={parallel})")
    print("=" * 80)
    
    results = []
    sizes = [1000, 5000, 10000, 50000]
    
    for size in sizes:
        x, y = generate_genomic_data(size, seed=42)
        
        def run():
            fastlowess.smooth(
                x, y,
                fraction=0.1,
                iterations=3,
                delta=100.0,
                scaling_method="mar",
                boundary_policy="noboundary",
                parallel=parallel
            )
        
        result = run_benchmark(f"genomic_{size}", size, run, iterations)
        results.append(result)
        print(f"  genomic_{size}: {result.mean_time_ms:.4f} ms ± {result.std_time_ms:.4f} ms")
    
    return results


def benchmark_pathological(iterations: int = 10, parallel: bool = True) -> List[BenchmarkResult]:
    """Benchmark with pathological/edge case data."""
    print("\n" + "=" * 80)
    print(f"PATHOLOGICAL (Parallel={parallel})")
    print("=" * 80)
    
    results = []
    size = 5000
    
    # Clustered
    x_clustered, y_clustered = generate_clustered_data(size, seed=42)
    result = run_benchmark(
        "clustered", size,
        lambda: fastlowess.smooth(
            x_clustered, y_clustered,
            fraction=0.3, iterations=2,
            scaling_method="mar", boundary_policy="noboundary", parallel=parallel
        ),
        iterations
    )
    results.append(result)
    print(f"  clustered: {result.mean_time_ms:.4f} ms ± {result.std_time_ms:.4f} ms")
    
    # High noise
    x_noisy, y_noisy = generate_high_noise_data(size, seed=42)
    result = run_benchmark(
        "high_noise", size,
        lambda: fastlowess.smooth(
            x_noisy, y_noisy,
            fraction=0.5, iterations=5,
            scaling_method="mar", boundary_policy="noboundary", parallel=parallel
        ),
        iterations
    )
    results.append(result)
    print(f"  high_noise: {result.mean_time_ms:.4f} ms ± {result.std_time_ms:.4f} ms")
    
    # Extreme outliers
    x_outlier, y_outlier = generate_outlier_data(size, seed=42)
    result = run_benchmark(
        "extreme_outliers", size,
        lambda: fastlowess.smooth(
            x_outlier, y_outlier,
            fraction=0.2, iterations=10,
            scaling_method="mar", boundary_policy="noboundary", parallel=parallel
        ),
        iterations
    )
    results.append(result)
    print(f"  extreme_outliers: {result.mean_time_ms:.4f} ms ± {result.std_time_ms:.4f} ms")
    
    # Constant y
    x_const = np.arange(size, dtype=float)
    y_const = np.full(size, 5.0)
    result = run_benchmark(
        "constant_y", size,
        lambda: fastlowess.smooth(
            x_const, y_const,
            fraction=0.2, iterations=2,
            scaling_method="mar", boundary_policy="noboundary", parallel=parallel
        ),
        iterations
    )
    results.append(result)
    print(f"  constant_y: {result.mean_time_ms:.4f} ms ± {result.std_time_ms:.4f} ms")
    
    return results


# ============================================================================
# Main Entry Point
# ============================================================================

def run_suite(parallel: bool, output_filename: str):
    print("\n" + "=" * 80)
    print(f"FASTLOWESS BENCHMARK SUITE (Parallel={parallel})")
    print(f"Output: {output_filename}")
    print("=" * 80)
    
    iterations = 25  # Reduced from 50 to save time double running
    all_results: Dict[str, List[BenchmarkResult]] = {}
    
    # Run all benchmark categories
    all_results["scalability"] = benchmark_scalability(iterations, parallel)
    all_results["fraction"] = benchmark_fraction(iterations, parallel)
    all_results["iterations"] = benchmark_iterations(iterations, parallel)
    all_results["delta"] = benchmark_delta(iterations, parallel)
    all_results["financial"] = benchmark_financial(iterations, parallel)
    all_results["scientific"] = benchmark_scientific(iterations, parallel)
    all_results["genomic"] = benchmark_genomic(iterations, parallel)
    all_results["pathological"] = benchmark_pathological(iterations, parallel)

    # Convert to JSON-serializable format
    output = {}
    for category, results in all_results.items():
        output[category] = [asdict(r) for r in results]
    
    # Save to output directory
    script_dir = Path(__file__).resolve().parent
    benchmarks_dir = script_dir.parent
    out_dir = benchmarks_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    out_path = out_dir / output_filename
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Results saved to {out_path}")
    print("=" * 80)


def main():
    # Run Parallel (Standard)
    run_suite(parallel=True, output_filename="fastlowess_benchmark.json")

    # Run Serial
    run_suite(parallel=False, output_filename="fastlowess_benchmark_serial.json")


if __name__ == "__main__":
    main()
