"""Download the public-domain books the workloads are built from (Project Gutenberg).

    python workloads/fetch_sources.py            # -> workloads/sources/pg<id>.txt

Pride and Prejudice, Moby Dick, Sherlock Holmes, Frankenstein, A Tale of Two
Cities, Alice in Wonderland, Dracula, The Picture of Dorian Gray. Texts can
change slightly upstream; workloads_meta.json records the checksums of the
prompt files these sources produced for the published runs.
"""
import pathlib
import urllib.request

IDS = [1342, 2701, 1661, 84, 98, 11, 345, 174]
out = pathlib.Path(__file__).parent / "sources"
out.mkdir(exist_ok=True)
for i in IDS:
    dest = out / f"pg{i}.txt"
    if not dest.exists():
        urllib.request.urlretrieve(f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt", dest)
    print(dest, dest.stat().st_size)
