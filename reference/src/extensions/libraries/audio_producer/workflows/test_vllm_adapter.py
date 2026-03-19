#!/usr/bin/env python3
"""Quick test of IndexTTS vLLM adapter interface."""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent))

from src.extensions.libraries.audio_producer.workflows.indextts_vllm_adapter import IndexTTSVLLMAdapter


def test_adapter_interface():
    """Test adapter can be imported and has correct interface."""
    print("=" * 50)
    print("IndexTTS vLLM Adapter Interface Test")
    print("=" * 50)

    # Create adapter
    print("\n[1/5] Creating adapter...")
    adapter = IndexTTSVLLMAdapter()
    print("  ✓ Adapter created")

    # Check attributes
    print("\n[2/5] Checking attributes...")
    assert hasattr(adapter, 'num_instances'), "Missing num_instances"
    assert adapter.num_instances == 1, f"Expected num_instances=1, got {adapter.num_instances}"
    assert hasattr(adapter, 'working_ports'), "Missing working_ports property"
    assert hasattr(adapter, 'generate_audio'), "Missing generate_audio method"
    assert hasattr(adapter, 'stop_all'), "Missing stop_all method"
    assert hasattr(adapter, 'cleanup'), "Missing cleanup method"
    assert hasattr(adapter, 'restart_hung_instance'), "Missing restart_hung_instance method"
    print("  ✓ All required attributes present")

    # Check working_ports
    print("\n[3/5] Checking working_ports...")
    ports = adapter.working_ports
    assert ports == {1}, f"Expected working_ports={{1}}, got {ports}"
    print("  ✓ working_ports = {1}")

    # Check method signatures
    print("\n[4/5] Checking method signatures...")
    import inspect
    sig = inspect.signature(adapter.generate_audio)
    params = list(sig.parameters.keys())
    assert 'worker_id' in params, "Missing worker_id parameter"
    assert 'text' in params, "Missing text parameter"
    assert 'voice_id' in params, "Missing voice_id parameter"
    assert 'emo_vector' in params, "Missing emo_vector parameter"
    assert 'reference_audio' in params, "Missing reference_audio parameter"
    print("  ✓ generate_audio has correct signature")

    sig = inspect.signature(adapter.restart_hung_instance)
    params = list(sig.parameters.keys())
    assert 'worker_id' in params, "Missing worker_id in restart_hung_instance"
    print("  ✓ restart_hung_instance has correct signature")

    # Check stats
    print("\n[5/5] Checking stats...")
    stats = adapter.get_stats()
    assert 'total_requests' in stats, "Missing total_requests in stats"
    assert 'total_chars' in stats, "Missing total_chars in stats"
    assert 'elapsed_time_sec' in stats, "Missing elapsed_time_sec in stats"
    assert 'chars_per_sec' in stats, "Missing chars_per_sec in stats"
    print("  ✓ Stats has correct fields")

    print("\n" + "=" * 50)
    print("All interface tests passed!")
    print("=" * 50)
    print("\nNote: Full initialization test requires:")
    print("  - modelscope package (for vLLM IndexTTS)")
    print("  - vLLM IndexTTS models downloaded")
    print("  - GPU memory for model loading")
    print("\nThe adapter interface is compatible with MultiIndexTTSManager.")
    return 0


if __name__ == "__main__":
    sys.exit(test_adapter_interface())
