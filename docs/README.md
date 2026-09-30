# TVRL - project page

Static site: every path is relative and every clip is bundled, so it runs from
any directory with no build step and no network access.

```bash
python3 tools/serve.py          # http://localhost:8000
```

Use that rather than `python3 -m http.server`: it is threaded and serves byte
ranges, which the filmstrip and the row scrubbers need. GitHub Pages and other
static hosts already do both.

```
index.html  site.css  tvrl.css  site.js  radar.js  data.js
videos/     all clips (864x480, H.264, no audio); the gallery shows every
            comparison from the paper (Figs. 1, 4, 6) as synced rows
posters/    first-frame stills
figs/       pipeline and routing figures from the paper
fonts/      Archivo and IBM Plex Mono, SIL OFL 1.1
tools/      serve.py, stamp_version.py, build_data.py
```

## Still to fill in

* the Paper button (search `index.html` for `Release:`) once the arXiv id exists;
* a BibTeX block under Results, from the same arXiv entry.

## Rebuilding the data

`tools/build_data.py` copies the clips from the anonymized supplementary
package and a few extra TVRL generations, writes posters, renders the two paper
figures to PNG, and regenerates `data.js`. Numbers in the tables and the radar
chart are typed into `index.html` and `radar.js` from the paper; update them
there if the paper changes. `index.html` loads assets as `name?v=N`; raise N
with `tools/stamp_version.py` after changing them on a host that caches.

Layout, styles, and the comparison-row player are adapted from the PDMD page.
