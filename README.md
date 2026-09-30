# TVRL: Token-Level Video Reinforcement Learning

**Project page:** coming soon · **Paper:** coming soon · **Code:** coming soon

TVRL derives token-level credit for video GRPO from the reward being optimized. A frozen
vision-language model scores each rollout by the teacher-forced likelihood of prompt-derived
yes/no checks, and the magnitude of the same likelihood's gradient with respect to the video
input routes the group-relative advantage to the video tokens that score depends on.

## Repository layout

```
docs/    project page (static site; served by GitHub Pages from /docs)
```

The training and evaluation code will be added here at release.

## Project page

```bash
cd docs && python3 tools/serve.py      # http://localhost:8000
```

See `docs/README.md` for what to fill in before the page goes public.

## Citation

BibTeX will be added once the paper is available.
