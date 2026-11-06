"""Indexing tests: a real temporary ChromaDB with mock embeddings; no model or API key."""
from dataclasses import replace
import hashlib
import logging
from pathlib import Path
import runpy
import subprocess
import sys
import unicodedata

import chromadb
from chromadb.api.models.Collection import Collection
import pymupdf
import pytest

from rag_hackathon import config, ingest as ingest_module, retrieval, vector_store
from rag_hackathon.chunking import chunk_records, chunk_text
from rag_hackathon.documents import load_documents
from rag_hackathon.ingest import ingest

COLLECTION = "test_documents"
REPO_ROOT = Path(__file__).resolve().parents[1]


def fake_embed(texts: list[str]) -> list[list[float]]:
    vectors = []
    for text in texts:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vectors.append([byte / 255 + 0.01 for byte in digest[:8]])
    return vectors


def use_settings(monkeypatch, tmp_path, **overrides):
    base = {
        "chroma_persist_dir": str(tmp_path / "chroma"),
        "chroma_collection": COLLECTION,
        "embedding_model": "mock-model",
        "chunk_size": 200,
        "chunk_overlap": 20,
        "max_cosine_distance": None,
    }
    updated = replace(config.settings, **{**base, **overrides})
    for module in (ingest_module, retrieval, vector_store):
        monkeypatch.setattr(module, "settings", updated)


@pytest.fixture
def index(monkeypatch, tmp_path):
    use_settings(monkeypatch, tmp_path)
    monkeypatch.setattr(ingest_module, "embed_texts", fake_embed)
    monkeypatch.setattr(retrieval, "embed_texts", fake_embed)
    return tmp_path


@pytest.fixture
def docs(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    return root


def write_txt(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stored(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "chroma"))
    return client, client.get_collection(COLLECTION)


# --- Duplicate filenames and chunk IDs ---

def test_same_filename_in_different_folders_keeps_both(index, docs):
    write_txt(docs, "2023/policy.txt", "old policy text")
    write_txt(docs, "2024/policy.txt", "new policy text")

    assert ingest(str(docs)) == 2

    _, collection = stored(index)
    result = collection.get(include=["documents", "metadatas"])
    assert sorted(result["ids"]) == ["2023/policy.txt::p1::c0", "2024/policy.txt::p1::c0"]
    assert sorted(result["documents"]) == ["new policy text", "old policy text"]
    assert {m["source"] for m in result["metadatas"]} == {"policy.txt"}


def test_chunk_ids_are_unique_and_stored_in_metadata(index, docs):
    write_txt(docs, "a/doc.txt", "\n".join(f"paragraph {i} " + "x " * 40 for i in range(10)))
    write_txt(docs, "b/doc.txt", "\n".join(f"paragraph {i} " + "y " * 40 for i in range(10)))

    ingest(str(docs))

    _, collection = stored(index)
    result = collection.get(include=["metadatas"])
    assert len(result["ids"]) > 2
    assert len(result["ids"]) == len(set(result["ids"]))
    assert all(m["chunk_id"] == id_ for id_, m in zip(result["ids"], result["metadatas"]))


def test_chunk_ids_are_stable_across_runs(index, docs):
    write_txt(docs, "a/doc.txt", "stable text")
    ingest(str(docs))
    ingest(str(docs))

    _, collection = stored(index)
    assert collection.get()["ids"] == ["a/doc.txt::p1::c0"]


def test_duplicate_chunk_ids_are_rejected_before_writing(index):
    metadata = {"chunk_id": "same"}
    chunks = [{"text": "a", "metadata": metadata}, {"text": "b", "metadata": metadata}]

    with pytest.raises(ValueError, match="Duplicate chunk IDs"):
        vector_store.upsert_chunks(chunks, fake_embed(["a", "b"]), chunk_size=200, chunk_overlap=20, documents_root="docs")

    assert vector_store.get_collection() is None


# --- Collection metadata ---

def test_collection_metadata_records_index_settings(index, docs):
    write_txt(docs, "doc.txt", "text")

    ingest(str(docs))

    _, collection = stored(index)
    assert collection.metadata["embedding_model"] == "mock-model"
    assert collection.metadata["chunk_size"] == 200
    assert collection.metadata["chunk_overlap"] == 20
    assert collection.metadata["hnsw:space"] == "cosine"


# --- Rebuild ---

def test_rebuild_with_different_chunk_size_replaces_all_chunks(monkeypatch, index, docs):
    write_txt(docs, "doc.txt", " ".join(f"word{i}" for i in range(300)))
    ingest(str(docs))
    _, before = stored(index)
    old_count = before.count()

    use_settings(monkeypatch, index, chunk_size=100, chunk_overlap=10)
    new_count = ingest(str(docs), rebuild=True)

    client, after = stored(index)
    assert new_count > old_count
    assert after.count() == new_count
    assert after.metadata["chunk_size"] == 100
    assert after.metadata["chunk_overlap"] == 10
    assert after.metadata["hnsw:space"] == "cosine"
    assert max(len(d) for d in after.get()["documents"]) <= 100
    assert [c.name for c in client.list_collections()] == [COLLECTION]


def test_rebuild_removes_chunks_of_deleted_files(index, docs):
    write_txt(docs, "keep.txt", "kept document")
    write_txt(docs, "remove.txt", "removed document")
    ingest(str(docs))
    (docs / "remove.txt").unlink()

    ingest(str(docs), rebuild=True)

    _, collection = stored(index)
    assert collection.get()["ids"] == ["keep.txt::p1::c0"]


def test_rebuilt_collection_is_searchable(index, docs):
    write_txt(docs, "doc.txt", "searchable text")
    ingest(str(docs))
    ingest(str(docs), rebuild=True)

    hits = retrieval.retrieve("searchable text")

    assert hits[0]["metadata"]["chunk_id"] == "doc.txt::p1::c0"
    assert hits[0]["distance"] == pytest.approx(0.0, abs=1e-5)


def test_rebuild_discards_stale_staging_collection(index, docs):
    write_txt(docs, "doc.txt", "text")
    client = chromadb.PersistentClient(path=str(index / "chroma"))
    stale = client.create_collection(COLLECTION + vector_store.STAGING_SUFFIX)
    stale.upsert(ids=["stale"], documents=["stale"], embeddings=[fake_embed(["stale"])[0]])

    ingest(str(docs), rebuild=True)

    _, collection = stored(index)
    assert collection.get()["ids"] == ["doc.txt::p1::c0"]


# --- Protection against accidental data loss ---

def test_normal_ingest_refuses_changed_chunk_settings(monkeypatch, index, docs):
    write_txt(docs, "doc.txt", "original text")
    ingest(str(docs))

    use_settings(monkeypatch, index, chunk_size=100, chunk_overlap=10)
    with pytest.raises(ValueError, match="--rebuild"):
        ingest(str(docs))

    _, collection = stored(index)
    assert collection.count() == 1
    assert collection.metadata["chunk_size"] == 200


def test_normal_ingest_refuses_changed_embedding_model(monkeypatch, index, docs):
    write_txt(docs, "doc.txt", "original text")
    ingest(str(docs))

    use_settings(monkeypatch, index, embedding_model="another-model")
    with pytest.raises(ValueError, match="embedding_model"):
        ingest(str(docs))


def test_normal_ingest_never_removes_chunks(index, docs):
    write_txt(docs, "one.txt", "first")
    write_txt(docs, "two.txt", "second")
    ingest(str(docs))
    (docs / "two.txt").unlink()

    ingest(str(docs))

    _, collection = stored(index)
    assert collection.count() == 2


def test_rebuild_with_empty_folder_keeps_index(index, docs):
    write_txt(docs, "doc.txt", "text")
    ingest(str(docs))
    (docs / "doc.txt").unlink()

    with pytest.raises(ValueError, match="No readable"):
        ingest(str(docs), rebuild=True)

    _, collection = stored(index)
    assert collection.count() == 1


def test_rebuild_refuses_empty_chunk_list(index):
    with pytest.raises(ValueError, match="empty index"):
        vector_store.rebuild_collection([], [], chunk_size=200, chunk_overlap=20, documents_root="docs")


def test_embedding_failure_during_rebuild_keeps_index(monkeypatch, index, docs):
    write_txt(docs, "doc.txt", "text")
    ingest(str(docs))

    def fail(_):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr(ingest_module, "embed_texts", fail)
    with pytest.raises(RuntimeError, match="embedding failed"):
        ingest(str(docs), rebuild=True)

    _, collection = stored(index)
    assert collection.count() == 1


def test_write_failure_during_rebuild_keeps_index_and_cleans_staging(monkeypatch, index, docs):
    write_txt(docs, "doc.txt", "text")
    ingest(str(docs))

    def fail(*_):
        raise RuntimeError("write failed")

    monkeypatch.setattr(vector_store, "_write", fail)
    with pytest.raises(RuntimeError, match="write failed"):
        ingest(str(docs), rebuild=True)

    client, collection = stored(index)
    assert collection.count() == 1
    assert [c.name for c in client.list_collections()] == [COLLECTION]


def test_reading_does_not_create_a_collection(index):
    assert retrieval.retrieve("any question") == []
    assert vector_store.get_collection() is None


# --- Source and page metadata ---

def test_pdf_page_and_source_metadata(index, docs):
    pdf_path = docs / "reports" / "annual.pdf"
    pdf_path.parent.mkdir()
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "First page text")
        pdf.new_page()  # A blank page must not shift the following page numbers.
        pdf.new_page().insert_text((72, 72), "Third page text")
        pdf.save(pdf_path)

    records = load_documents(docs)

    assert [(r["metadata"]["page"], r["text"]) for r in records] == [
        (1, "First page text"),
        (3, "Third page text"),
    ]
    for record in records:
        assert record["metadata"]["source"] == "annual.pdf"
        assert record["metadata"]["source_path"] == "reports/annual.pdf"
        assert record["metadata"]["file_type"] == "pdf"

    ingest(str(docs))
    _, collection = stored(index)
    result = collection.get(ids=["reports/annual.pdf::p3::c0"], include=["documents", "metadatas"])
    assert result["documents"] == ["Third page text"]
    assert result["metadatas"][0]["page"] == 3


def test_txt_source_metadata_with_arabic_filename(docs):
    write_txt(docs, "سياسات/دليل.txt", "نص عربي")

    metadata = load_documents(docs)[0]["metadata"]

    assert metadata["source"] == "دليل.txt"
    assert metadata["source_path"] == "سياسات/دليل.txt"
    assert metadata["page"] == 1
    assert metadata["file_type"] == "txt"


# --- Arabic chunk boundaries ---

ARABIC_WORD = "الْمَعْلُومَاتِ"
ARABIC_DIACRITICS = "ًٌٍَُِّْ"


@pytest.mark.parametrize("size,overlap", [(50, 0), (50, 10), (97, 13), (200, 30)])
def test_long_arabic_paragraph_is_split_on_word_boundaries(size, overlap):
    text = " ".join([ARABIC_WORD] * 200)

    chunks = chunk_text(text, size=size, overlap=overlap)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= size
        assert set(chunk.split()) == {ARABIC_WORD}
        assert chunk[0] not in ARABIC_DIACRITICS


def test_arabic_paragraph_overlap_tail_keeps_whole_words():
    paragraphs = [" ".join(f"{ARABIC_WORD}{n}" for _ in range(6)) for n in range(6)]
    words = set(" ".join(paragraphs).split())

    chunks = chunk_text("\n".join(paragraphs), size=150, overlap=40)

    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 150
        assert set(chunk.split()) <= words


def test_arabic_text_is_not_lost_or_altered_by_chunking():
    words = [f"كلمة{i}" for i in range(500)]

    chunks = chunk_text(" ".join(words), size=120, overlap=20)

    seen = [word for chunk in chunks for word in chunk.split()]
    assert set(seen) == set(words)


def test_chunk_records_keeps_arabic_page_metadata():
    records = [{
        "text": " ".join([ARABIC_WORD] * 50),
        "metadata": {"source": "دليل.pdf", "source_path": "سياسات/دليل.pdf", "page": 7, "file_type": "pdf"},
    }]

    chunks = chunk_records(records, size=60, overlap=10)

    assert len(chunks) > 1
    assert {c["metadata"]["page"] for c in chunks} == {7}
    assert chunks[1]["metadata"]["chunk_id"] == "سياسات/دليل.pdf::p7::c1"


# --- Rebuild failure handling ---

def test_failed_swap_removes_staging_and_keeps_old_index(monkeypatch, index, docs):
    write_txt(docs, "doc.txt", "original")
    ingest(str(docs))
    write_txt(docs, "doc.txt", "replacement")
    real_modify = Collection.modify

    def fail_promoting_staging(self, *args, **kwargs):
        if self.name.endswith(vector_store.STAGING_SUFFIX):
            raise RuntimeError("swap failed")
        return real_modify(self, *args, **kwargs)

    monkeypatch.setattr(Collection, "modify", fail_promoting_staging)
    with pytest.raises(RuntimeError, match="swap failed"):
        ingest(str(docs), rebuild=True)

    client, collection = stored(index)
    assert collection.get()["documents"] == ["original"]
    assert [c.name for c in client.list_collections()] == [COLLECTION]


def test_failed_rollback_keeps_both_copies_and_reports_interruption(monkeypatch, index, docs):
    write_txt(docs, "doc.txt", "original")
    ingest(str(docs))
    real_modify = Collection.modify

    def fail_after_backup(self, *args, **kwargs):
        if self.name.endswith((vector_store.STAGING_SUFFIX, vector_store.BACKUP_SUFFIX)):
            raise RuntimeError("rename failed")
        return real_modify(self, *args, **kwargs)

    monkeypatch.setattr(Collection, "modify", fail_after_backup)
    with pytest.raises(RuntimeError, match="rename failed"):
        ingest(str(docs), rebuild=True)
    monkeypatch.setattr(Collection, "modify", real_modify)

    client = chromadb.PersistentClient(path=str(index / "chroma"))
    assert {c.name for c in client.list_collections()} == {
        COLLECTION + vector_store.BACKUP_SUFFIX,
        COLLECTION + vector_store.STAGING_SUFFIX,
    }
    with pytest.raises(vector_store.InterruptedRebuildError):
        vector_store.get_collection()


def test_incomplete_staging_is_never_promoted(monkeypatch, index, docs):
    write_txt(docs, "doc.txt", "first paragraph\n" + "second " * 200)
    ingest(str(docs))
    real_write = vector_store._write

    def write_one(collection, ids, chunks, embeddings, batch_size):
        real_write(collection, ids[:1], chunks[:1], embeddings[:1], batch_size)

    monkeypatch.setattr(vector_store, "_write", write_one)
    with pytest.raises(RuntimeError, match="incomplete"):
        ingest(str(docs), rebuild=True)

    client, collection = stored(index)
    assert collection.count() > 1
    assert [c.name for c in client.list_collections()] == [COLLECTION]


def test_cleanup_failure_does_not_hide_the_original_error(monkeypatch, index, docs, caplog):
    write_txt(docs, "doc.txt", "text")
    ingest(str(docs))

    def fail_write(*_):
        raise RuntimeError("original write error")

    def fail_delete(*_):
        raise OSError("cleanup error")

    monkeypatch.setattr(vector_store, "_write", fail_write)
    monkeypatch.setattr(chromadb.api.client.Client, "delete_collection", fail_delete)
    with pytest.raises(RuntimeError, match="original write error"):
        ingest(str(docs), rebuild=True)

    assert "cleanup error" in caplog.text


def test_backup_delete_failure_after_swap_is_a_warning(monkeypatch, index, docs, caplog):
    write_txt(docs, "doc.txt", "new text")
    ingest(str(docs))
    real_delete = chromadb.api.client.Client.delete_collection

    def fail_backup_delete(self, name, *args, **kwargs):
        if name.endswith(vector_store.BACKUP_SUFFIX):
            raise OSError("cannot delete backup")
        return real_delete(self, name, *args, **kwargs)

    monkeypatch.setattr(chromadb.api.client.Client, "delete_collection", fail_backup_delete)

    assert ingest(str(docs), rebuild=True) == 1
    assert "cannot delete backup" in caplog.text


def test_rebuild_replaces_stale_backup_next_to_live_collection(index, docs):
    write_txt(docs, "doc.txt", "text")
    ingest(str(docs))
    client = chromadb.PersistentClient(path=str(index / "chroma"))
    client.create_collection(COLLECTION + vector_store.BACKUP_SUFFIX)

    ingest(str(docs), rebuild=True)

    client, collection = stored(index)
    assert collection.get()["ids"] == ["doc.txt::p1::c0"]
    assert [c.name for c in client.list_collections()] == [COLLECTION]


# --- Interrupted rebuild: backup without a live collection ---

def interrupt_rebuild(index):
    client = chromadb.PersistentClient(path=str(index / "chroma"))
    client.get_collection(COLLECTION).modify(name=COLLECTION + vector_store.BACKUP_SUFFIX)


def test_questions_fail_clearly_after_interrupted_rebuild(index, docs):
    write_txt(docs, "doc.txt", "text")
    ingest(str(docs))
    interrupt_rebuild(index)

    with pytest.raises(vector_store.InterruptedRebuildError, match="--rebuild"):
        retrieval.retrieve("any question")


def test_normal_ingest_refuses_after_interrupted_rebuild(index, docs):
    write_txt(docs, "doc.txt", "text")
    ingest(str(docs))
    interrupt_rebuild(index)

    with pytest.raises(vector_store.InterruptedRebuildError):
        ingest(str(docs))

    client = chromadb.PersistentClient(path=str(index / "chroma"))
    assert [c.name for c in client.list_collections()] == [COLLECTION + vector_store.BACKUP_SUFFIX]


def test_rebuild_repairs_interrupted_rebuild_and_removes_backup(index, docs):
    write_txt(docs, "doc.txt", "text")
    ingest(str(docs))
    interrupt_rebuild(index)

    ingest(str(docs), rebuild=True)

    client, collection = stored(index)
    assert collection.count() == 1
    assert [c.name for c in client.list_collections()] == [COLLECTION]


def test_documented_restore_of_backup_recovers_index(index, docs):
    write_txt(docs, "doc.txt", "searchable text")
    ingest(str(docs))
    interrupt_rebuild(index)

    client = chromadb.PersistentClient(path=str(index / "chroma"))
    client.get_collection(COLLECTION + vector_store.BACKUP_SUFFIX).modify(name=COLLECTION)

    assert retrieval.retrieve("searchable text")[0]["metadata"]["chunk_id"] == "doc.txt::p1::c0"


# --- Validation and batching ---

def test_overlap_only_change_is_refused(monkeypatch, index, docs):
    write_txt(docs, "doc.txt", "text")
    ingest(str(docs))

    use_settings(monkeypatch, index, chunk_size=200, chunk_overlap=10)
    with pytest.raises(ValueError, match="chunk_overlap"):
        ingest(str(docs))


@pytest.mark.parametrize("write", [vector_store.upsert_chunks, vector_store.rebuild_collection])
def test_embedding_count_mismatch_is_rejected(index, write):
    chunks = [{"text": "a", "metadata": {"chunk_id": "a"}}]

    with pytest.raises(ValueError, match="exactly one embedding"):
        write(chunks, [], chunk_size=200, chunk_overlap=20, documents_root="docs")

    assert vector_store.get_collection() is None


def make_chunks(count):
    return [{"text": f"chunk {i}", "metadata": {"chunk_id": f"doc.txt::p1::c{i}"}} for i in range(count)]


def test_writes_are_split_into_batches(monkeypatch, index):
    sizes = []
    real_upsert = Collection.upsert

    def record(self, *args, ids, **kwargs):
        sizes.append(len(ids))
        return real_upsert(self, *args, ids=ids, **kwargs)

    monkeypatch.setattr(chromadb.api.client.Client, "get_max_batch_size", lambda self: 2)
    monkeypatch.setattr(Collection, "upsert", record)
    chunks = make_chunks(5)

    vector_store.upsert_chunks(chunks, fake_embed([c["text"] for c in chunks]), chunk_size=200, chunk_overlap=20, documents_root="docs")

    assert sizes == [2, 2, 1]
    assert vector_store.get_collection().count() == 5


def test_rebuild_above_real_chroma_batch_limit(index):
    limit = chromadb.PersistentClient(path=str(index / "chroma")).get_max_batch_size()
    chunks = make_chunks(limit + 1)
    embeddings = fake_embed([c["text"] for c in chunks])

    assert vector_store.rebuild_collection(chunks, embeddings, chunk_size=200, chunk_overlap=20, documents_root="docs") == limit + 1
    assert vector_store.get_collection().count() == limit + 1


def test_rebuilt_index_survives_process_restart(index):
    chunks = make_chunks(1200)  # Above Chroma's default in-memory HNSW sync threshold.
    embeddings = fake_embed([c["text"] for c in chunks])
    vector_store.rebuild_collection(chunks, embeddings, chunk_size=200, chunk_overlap=20, documents_root="docs")

    code = (
        "import sys, chromadb\n"
        "c = chromadb.PersistentClient(path=sys.argv[1]).get_collection(sys.argv[2])\n"
        "q = c.query(query_embeddings=[[float(x) for x in sys.argv[3].split(',')]], n_results=1)\n"
        "print(c.count(), q['ids'][0][0])\n"
    )
    vector = ",".join(str(x) for x in embeddings[777])
    result = subprocess.run(
        [sys.executable, "-c", code, str(index / "chroma"), COLLECTION, vector],
        capture_output=True, text=True, check=True,
    )

    assert result.stdout.split() == ["1200", "doc.txt::p1::c777"]


# --- Command line ---

def run_cli(monkeypatch, *args):
    calls = []
    def fake_ingest(folder, *, rebuild=False, ocr_files=()):
        calls.append((folder, rebuild, list(ocr_files)))

    monkeypatch.setattr(ingest_module, "ingest", fake_ingest)
    monkeypatch.setattr(sys, "argv", ["ingest.py", *args])
    runpy.run_path(str(REPO_ROOT / "scripts" / "ingest.py"), run_name="__main__")
    return calls


def test_cli_does_not_rebuild_by_default(monkeypatch):
    assert run_cli(monkeypatch) == [("data/raw", False, [])]


def test_cli_rebuild_flag_requests_rebuild(monkeypatch):
    assert run_cli(monkeypatch, "--rebuild") == [("data/raw", True, [])]


def test_cli_indexes_requested_folder(monkeypatch):
    assert run_cli(monkeypatch, "--folder", "data/sample") == [("data/sample", False, [])]


def test_cli_passes_requested_ocr_files(monkeypatch):
    calls = run_cli(monkeypatch, "--ocr-file", "a.pdf", "--ocr-file", "b/c.pdf")

    assert calls == [("data/raw", False, ["a.pdf", "b/c.pdf"])]


# --- Extraction warnings and path normalization ---

def test_pages_without_text_are_reported(docs, caplog):
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Text page")
        pdf.new_page()
        pdf.new_page().insert_text((72, 72), "Another text page")
        pdf.new_page()
        pdf.save(docs / "mixed.pdf")

    with caplog.at_level(logging.WARNING, logger="rag_hackathon.documents"):
        records = load_documents(docs)

    assert [r["metadata"]["page"] for r in records] == [1, 3]
    assert "mixed.pdf" in caplog.text
    assert "[2, 4]" in caplog.text


def test_fully_extracted_pdf_has_no_warning(docs, caplog):
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Text page")
        pdf.save(docs / "clean.pdf")

    with caplog.at_level(logging.WARNING, logger="rag_hackathon.documents"):
        load_documents(docs)

    assert caplog.text == ""


def test_source_path_is_normalized_to_nfc(docs):
    composed = "أسئلة.txt"
    write_txt(docs, unicodedata.normalize("NFD", composed), "text")

    assert load_documents(docs)[0]["metadata"]["source_path"] == composed


# --- Chunk overlap and whitespace ---

@pytest.mark.parametrize("separator", ["\t", " ", " ", "\n"], ids=["tab", "nbsp", "thin-space", "newline"])
def test_words_stay_whole_with_any_whitespace(separator):
    words = [f"كلمة{i}" for i in range(300)]

    chunks = chunk_text(separator.join(words), size=100, overlap=20)

    assert {word for chunk in chunks for word in chunk.split()} == set(words)
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_single_word_lines_are_not_cut_when_overlap_is_large():
    words = ["كلمة2416", "كلمة3120", "كلمة4501", "كلمة1234"]

    chunks = chunk_text("\n".join(words), size=23, overlap=19)

    assert {word for chunk in chunks for word in chunk.split()} == set(words)


def test_no_overlap_reproduces_every_word_in_order():
    words = [f"كلمة{i}" for i in range(500)]

    chunks = chunk_text(" ".join(words), size=120, overlap=0)

    assert " ".join(chunks).split() == words


def test_long_paragraph_neighbours_share_trailing_words():
    words = [f"كلمة{i}" for i in range(300)]

    chunks = chunk_text(" ".join(words), size=100, overlap=30)

    assert len(chunks) > 2
    for first, second in zip(chunks, chunks[1:]):
        shared = len(set(first.split()) & set(second.split()))
        assert shared > 0
        assert first.split()[-shared:] == second.split()[:shared]


def test_unbroken_text_keeps_exact_overlap():
    text = "".join(chr(0x0621 + i % 26) for i in range(1200))

    chunks = chunk_text(text, size=500, overlap=100)

    assert len(chunks) == 3
    for first, second in zip(chunks, chunks[1:]):
        assert first[-100:] == second[:100]


def test_paragraph_neighbours_share_overlap():
    paragraphs = [" ".join(f"فقرة{p}كلمة{i}" for i in range(6)) for p in range(8)]

    chunks = chunk_text("\n".join(paragraphs), size=150, overlap=40)

    assert len(chunks) > 2
    for first, second in zip(chunks, chunks[1:]):
        assert set(first.split()) & set(second.split())


# --- One documents root per collection ---

def test_collection_metadata_records_documents_root(index, docs):
    write_txt(docs, "doc.txt", "text")

    ingest(str(docs))

    _, collection = stored(index)
    assert collection.metadata["documents_root"] == docs.resolve().as_posix()


def test_second_documents_root_is_refused_and_keeps_chunks(index, docs, tmp_path):
    other = tmp_path / "other"
    write_txt(docs, "policy.txt", "first source")
    write_txt(other, "policy.txt", "second source")
    ingest(str(docs))

    with pytest.raises(ValueError, match="documents_root"):
        ingest(str(other))

    _, collection = stored(index)
    assert collection.get()["documents"] == ["first source"]


def test_same_root_spelled_differently_is_accepted(monkeypatch, index, docs, tmp_path):
    write_txt(docs, "doc.txt", "text")
    ingest(str(docs))
    monkeypatch.chdir(tmp_path)

    assert ingest("raw") == 1
    assert ingest(str(docs / "." )) == 1


def test_rebuild_can_switch_documents_root(index, docs, tmp_path):
    other = tmp_path / "other"
    write_txt(docs, "old.txt", "old source")
    write_txt(other, "new.txt", "new source")
    ingest(str(docs))

    ingest(str(other), rebuild=True)

    _, collection = stored(index)
    assert collection.get()["ids"] == ["new.txt::p1::c0"]
    assert collection.metadata["documents_root"] == other.resolve().as_posix()


# --- Embedding model check at question time ---

def test_questions_are_refused_with_a_different_embedding_model(monkeypatch, index, docs):
    write_txt(docs, "doc.txt", "text")
    ingest(str(docs))
    use_settings(monkeypatch, index, embedding_model="another-model")

    def must_not_embed(_):
        raise AssertionError("the question must not be embedded")

    monkeypatch.setattr(retrieval, "embed_texts", must_not_embed)
    with pytest.raises(vector_store.EmbeddingModelMismatchError, match="--rebuild"):
        retrieval.retrieve("any question")


# --- Stale chunks warning ---

def test_normal_ingest_warns_about_stale_chunks(index, docs, caplog):
    write_txt(docs, "one.txt", "first")
    write_txt(docs, "two.txt", "second")
    ingest(str(docs))
    (docs / "two.txt").unlink()

    with caplog.at_level(logging.WARNING, logger="rag_hackathon.ingest"):
        ingest(str(docs))

    assert "1 stale chunks" in caplog.text
    assert "--rebuild" in caplog.text


def test_no_stale_warning_when_index_matches_documents(index, docs, caplog):
    write_txt(docs, "doc.txt", "text")
    ingest(str(docs))

    with caplog.at_level(logging.WARNING, logger="rag_hackathon.ingest"):
        ingest(str(docs))

    assert "stale" not in caplog.text


def test_no_stale_warning_after_rebuild(index, docs, caplog):
    write_txt(docs, "one.txt", "first")
    write_txt(docs, "two.txt", "second")
    ingest(str(docs))
    (docs / "two.txt").unlink()

    with caplog.at_level(logging.WARNING, logger="rag_hackathon.ingest"):
        ingest(str(docs), rebuild=True)

    assert "stale" not in caplog.text
