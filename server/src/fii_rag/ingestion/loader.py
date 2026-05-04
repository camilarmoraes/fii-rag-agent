"""Carregamento de PDF: texto + páginas via PyMuPDF, markdown via pymupdf4llm."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader

from server.src.fii_rag.chunking.base import LoadedDocument, PageText


class PdfLoader:
    """Carrega o PDF em memória produzindo `LoadedDocument`.

    - `pages`: extraído via `PyMuPDFLoader` (preserva número de página)
    - `markdown`: extraído via `pymupdf4llm.to_markdown` (heading-aware);
      em caso de falha, fica string vazia → `DocAwareChunker` cai em recursive
    """

    def load(self, path: str) -> LoadedDocument:
        path_obj = Path(path)
        loader = PyMuPDFLoader(str(path_obj))
        lc_docs = loader.load()
        pages = [
            PageText(
                page_number=int(d.metadata.get("page", i)) + 1,
                text=d.page_content,
            )
            for i, d in enumerate(lc_docs)
        ]
        full_text = "\n\n".join(p.text for p in pages)

        markdown = ""
        try:
            import pymupdf4llm

            markdown = pymupdf4llm.to_markdown(str(path_obj)) or ""
        except Exception as e:  # noqa: BLE001
            print(f"[PdfLoader] pymupdf4llm.to_markdown falhou: {e}. Markdown vazio.")

        return LoadedDocument(
            full_text=full_text,
            pages=pages,
            markdown=markdown,
            total_pages=len(pages),
            source_path=str(path_obj),
        )
