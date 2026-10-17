"""Build the runtime-only ZIP from the verified standalone entrypoint."""
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(__file__).resolve().parents[1]
data=(ROOT/'submission/agent.py').read_bytes()
expected=json.loads((ROOT/'evidence/release.json').read_text())['agent_sha256']
assert hashlib.sha256(data).hexdigest()==expected,'Source changed; revalidate and update release metadata first.'
output=ROOT/'releases/Odyssey_submission.zip'
output.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('agent.py',data)
with zipfile.ZipFile(output) as z:assert z.read('agent.py')==data and z.namelist()==['agent.py']
print(output)
