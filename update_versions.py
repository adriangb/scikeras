"""Writes a list like:

    refs/heads/main,latest
    v1.0.0,stable
    v0.9.0,v0.9.0

To a file called versions.txt
"""

from json import dumps
import os

base = "refs"
branch_names = {"master": "latest"}
stable = None

versions = dict()

if os.path.exists(os.path.join(base, "heads")):
    for branch in sorted(os.listdir(os.path.join(base, "heads"))):
        if branch not in branch_names:
            continue
        bname = branch_names[branch]
        versions[bname] = "/".join((base, "heads", branch))
if os.path.exists(os.path.join(base, "tags")):
    tags = []
    for tag in os.listdir(os.path.join(base, "tags")):
        tags.append(tag)
    tags.sort(key=lambda s: map(int, s.strip("v").split('.')))
    for tag in tags[:-1]:
        versions[tag] = "/".join((base, "tags", tag))
    stable = "/".join((base, "tags", tags[-1]))
    versions["stable"] = stable

with open("versions.json", "w") as f:
    f.write(dumps(versions))

if stable:
    redirect = stable
else:
    try:
        b = next(k for k, v in branch_names.items() if v )
        redirect = "/".join((base, "heads", b))
    except StopIteration:
        redirect = None
if redirect:
    with open("index.html", "w") as f:
        f.write(f"""<meta http-equiv="refresh" content="0; URL='{redirect}/index.html'" />""")
