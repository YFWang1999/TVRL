# TVRL: Token-Level Video Reinforcement Learning

**[Project page](https://yfwang1999.github.io/TVRL/)** · **Paper:** coming soon · **Code:** coming soon

Yifan Wang<sup>1</sup>, Gordon Guocheng Qian<sup>†</sup>, Yanyu Li, Anil Kag, Yun Fu<sup>1</sup>  
<sup>1</sup>Northeastern University · <sup>†</sup>Corresponding author

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

See `docs/README.md` for what is still to be filled in.

## Citation

BibTeX will be added once the paper is available.
