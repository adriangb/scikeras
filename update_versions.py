"""This script is run from docs_deploy.yml

It writes JSON like this:

    {"latest": "refs/heads/main"}

To a file called versions.json at the root
of the docs branch.

It will also update index.html to point to the
latest stable (released) version, or if no released versions
exist, the the main branch's version.
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
    tags.sort(key=lambda s: tuple(map(int, s.strip("v").split("."))))
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
        b = next(k for k, v in branch_names.items() if v)
        redirect = "/".join((base, "heads", b))
    except StopIteration:
        redirect = None
if redirect:
    with open("index.html", "w") as f:
        f.write(
            f"""<meta http-equiv="refresh" content="0; URL='{redirect}/index.html'" />"""
        )
