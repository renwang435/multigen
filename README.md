# The Sound of Simulation: Learning Multimodal Sim-to-Real Robot Policies with Generative Audio

[Renhao Wang](https://renwang435.github.io/), [Haoran Geng](https://geng-haoran.github.io/), [Tingle Li](https://tinglok.netlify.app/), [Feishi Wang](https://scholar.google.com/citations?user=eGG8hJgAAAAJ&hl=en), [Gopala Anumanchipalli](https://www2.eecs.berkeley.edu/Faculty/Homepages/gopala.html), [Boyi Li](https://sites.google.com/site/boyilics/home), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/), [Jitendra Malik](https://people.eecs.berkeley.edu/~malik/), [Alexei A. Efros](http://people.eecs.berkeley.edu/~efros/)

[[`arXiv`](https://arxiv.org/abs/2507.02864)] [[`BibTeX`](#Citing)]

## Code Structure

Our code release consists of two main sections. The [first section](https://github.com/renwang435/multigen/simulation/README.md) involves physics-based simulation for generating motion planned pouring trajectories. The [second section](https://github.com/renwang435/multigen/generation/README.md) involves training a video-to-audio diffusion model for synchronized pouring. This section also includes [inference code](https://github.com/renwang435/multigen/generation/README.md#L127) for generating audio tracks given the simulated video from the first section.

## <a name="Citing"></a>Citing MultiGen

```BibTeX
@inproceedings{
    wang2025the,
    title={The Sound of Simulation: Learning Multimodal Sim-to-Real Robot Policies with Generative Audio},
    author={Renhao Wang and Haoran Geng and Tingle Li and Philipp Wu and Feishi Wang and Gopala Anumanchipalli and Trevor Darrell and Boyi Li and Pieter Abbeel and Jitendra Malik and Alexei A Efros},
    booktitle={9th Annual Conference on Robot Learning},
    year={2025},
    url={https://openreview.net/forum?id=a9RXjOt5bU}
}
```


