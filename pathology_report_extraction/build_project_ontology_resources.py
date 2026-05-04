from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pathology_report_extraction.ontology.build_project_ontology_resources import *  # noqa: F401,F403
from pathology_report_extraction.ontology.build_project_ontology_resources import (
    _append_synonym_record,
    _append_xref,
    _clone_xrefs,
    _copy_lexicalizations,
    _merge_xrefs,
)

if __name__ == "__main__":
    from pathology_report_extraction.ontology.build_project_ontology_resources import main

    main()
