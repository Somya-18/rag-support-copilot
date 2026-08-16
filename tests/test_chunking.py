from kube_copilot.ingestion import chunk_markdown


def test_heading_aware_chunks_preserve_metadata_and_lines():
    raw = """---
title: Debug Pods
---
# Debugging

Intro paragraph.

## Inspect events

Use kubectl describe to inspect recorded events.
"""
    metadata, chunks = chunk_markdown(raw, "debug-pods.md", target=20)
    assert metadata["title"] == "Debug Pods"
    assert chunks
    assert any("Inspect events" in chunk.heading_path for chunk in chunks)
    assert all(chunk.line_start > 0 and chunk.line_end >= chunk.line_start for chunk in chunks)
    assert all(chunk.embedded_text.startswith("Debug Pods") for chunk in chunks)


def test_large_section_respects_maximum_tokens():
    raw = "# Long\n\n" + "word " * 1200
    _, chunks = chunk_markdown(raw, "long.md", maximum=100)
    assert len(chunks) > 1
    assert all(chunk.token_count <= 100 for chunk in chunks)
