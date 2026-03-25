import numpy as np
import pytest
from pathlib import Path
from unittest.mock import AsyncMock
from deep_research_engine.shared.multimodal import unify_speakers

@pytest.mark.asyncio
async def test_unify_speakers_logic(mocker, tmp_path):
    """
    Verify that unify_speakers correctly maps speaker IDs based on embedding similarity.
    """
    # 1. Mock extract_audio_snippet (to return True)
    # Since it's an async function, we use AsyncMock
    mock_snippet = mocker.patch(
        "deep_research_engine.shared.multimodal.extract_audio_snippet", 
        new_callable=AsyncMock,
        return_value=True
    )
    
    # 2. Mock extract_speaker_embedding (to return fake numpy arrays)
    # We want mf0:S1 and mf1:S2 to be similar (similarity >= 0.8)
    # and mf0:S3 to be different (similarity < 0.8)
    
    # Cosine similarity of [1,0,0] and [0.99, 0.01, 0.0] is very high (>0.99)
    # Cosine similarity of [1,0,0] and [0, 1, 0] is 0
    emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    emb2 = np.array([0.99, 0.01, 0.0], dtype=np.float32) 
    emb3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    
    # Map snippet filenames to their embeddings
    # The code uses snippet_path = snippet_dir / f"mf{mf_idx}_{speaker_id}.wav"
    embedding_map = {
        "mf0_S1.wav": emb1,
        "mf1_S2.wav": emb2,
        "mf0_S3.wav": emb3,
    }
    
    async def mock_emb(path):
        return embedding_map.get(path.name, np.random.rand(3).astype(np.float32))
    
    mocker.patch(
        "deep_research_engine.shared.multimodal.extract_speaker_embedding", 
        side_effect=mock_emb
    )
    
    # 3. Setup mega_file_results input
    # Each entry is (Path, dict)
    mega_file_results = [
        (
            Path("/fake/audio/mf0.wav"),
            {
                "segments": [
                    {"speaker": "S1", "start": 0.0, "duration": 10.0},
                    {"speaker": "S3", "start": 10.0, "duration": 10.0},
                ],
                "speaker_list": ["S1", "S3"]
            }
        ),
        (
            Path("/fake/audio/mf1.wav"),
            {
                "segments": [
                    {"speaker": "S2", "start": 0.0, "duration": 10.0},
                ],
                "speaker_list": ["S2"]
            }
        )
    ]
    
    output_dir = tmp_path / "output"
    
    # Call the function
    global_speaker_map = await unify_speakers(mega_file_results, output_dir)
    
    # 4. Verify results
    
    # - Two speakers from different mega-files with similar embeddings (similarity >= 0.8) 
    #   are mapped to the same Global_Speaker_ID.
    # mf0:S1 and mf1:S2 should have the same global ID
    assert global_speaker_map["mf0:S1"] == global_speaker_map["mf1:S2"]
    
    # - Speakers with different embeddings get unique IDs.
    # mf0:S3 should have a different ID
    assert global_speaker_map["mf0:S3"] != global_speaker_map["mf0:S1"]
    
    # - The global_speaker_map correctly maps the mf{idx}:{speaker_id} format to Global_Speaker_{n}.
    assert global_speaker_map["mf0:S1"].startswith("Global_Speaker_")
    assert global_speaker_map["mf0:S3"].startswith("Global_Speaker_")
    
    # Ensure all expected keys are in the map
    assert set(global_speaker_map.keys()) == {"mf0:S1", "mf0:S3", "mf1:S2"}
    
    # Ensure IDs are sequential starting from Global_Speaker_0
    all_ids = sorted(list(set(global_speaker_map.values())))
    assert all_ids == ["Global_Speaker_0", "Global_Speaker_1"]
    
    # Verify that snippets were "extracted" (mock called)
    assert mock_snippet.call_count == 3

@pytest.mark.asyncio
async def test_unify_speakers_skips_no_segments(mocker, tmp_path):
    """
    Verify that unify_speakers skips speakers that have no segments.
    """
    mocker.patch(
        "deep_research_engine.shared.multimodal.extract_audio_snippet", 
        new_callable=AsyncMock,
        return_value=True
    )
    
    mocker.patch(
        "deep_research_engine.shared.multimodal.extract_speaker_embedding", 
        new_callable=AsyncMock,
        return_value=np.array([1.0, 0.0, 0.0])
    )
    
    mega_file_results = [
        (
            Path("/fake/audio/mf0.wav"),
            {
                "segments": [], # No segments
                "speaker_list": ["S1"]
            }
        )
    ]
    
    output_dir = tmp_path / "output"
    global_speaker_map = await unify_speakers(mega_file_results, output_dir)
    
    # Should be empty
    assert global_speaker_map == {}
