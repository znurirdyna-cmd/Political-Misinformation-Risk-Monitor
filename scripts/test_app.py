"""Headless smoke test of the end-user Streamlit app.

Loads app.py and confirms the "Stories to review" page renders. (Full
click-through of the segmented-control list is left to browser testing —
Streamlit's AppTest cannot yet serialize st.segmented_control state.)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=300).run()
assert not at.exception, at.exception

titles = [t.value for t in at.title]
assert any("Overview" in t for t in titles), f"overview title missing: {titles}"

buttons = [b.label for b in at.button]
assert "View details" in buttons, "story list did not render"

print("APP OK — Articles for Review page rendered without errors")
