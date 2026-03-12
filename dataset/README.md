<div align="center">

# 📊 VisForm Dataset

### A Large-Scale Benchmark for Evaluating Generative Image Models Under Broad Distribution Shifts

<a href="https://arxiv.org/abs/2603.08064" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2603.08064-red?logo=arxiv" height="25" />
</a>
<a href="https://huggingface.co/datasets/ZexiJia/Visform" target="_blank">
    <img alt="Dataset" src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-VisForm-ffc107?color=ffc107&logoColor=white" height="25" />
</a>

</div>

---

## Overview

**VisForm** is a large-scale benchmark for evaluating generative image models under broad distribution shifts.

Unlike benchmarks centered mostly on photorealistic imagery, VisForm covers a much wider spectrum of visual content, including photography, painting, illustration, diagrams, scientific imagery, UI-like graphics, sensor-style images, and design elements.

It is designed for:

- Cross-domain generative model evaluation
- Image quality metric benchmarking
- Metric–human alignment analysis
- Quality, aesthetics, and safety assessment

🤗 **Download**: [https://huggingface.co/datasets/ZexiJia/Visform](https://huggingface.co/datasets/ZexiJia/Visform)

---

## Highlights

- **210,000** images
- **62** visual forms
- **12** representative generative models
- **14** perceptual dimensions
- At least **3 expert annotators** per image

---

## What Makes VisForm Useful?

VisForm is built for settings where many existing evaluation benchmarks and metrics become less reliable, especially on:

- Artistic imagery
- Symbolic or structured graphics
- Text-heavy layouts
- Scientific and medical visualizations
- Functional images such as depth maps and other sensor outputs

By explicitly covering these diverse forms, VisForm provides a stronger testbed for evaluating robustness beyond natural photos.

---

## Dataset Content

Each sample is associated with structured annotations such as:

- Visual form
- Source model
- Fine-grained artifact labels
- 5-point expert ratings

The benchmark focuses on three major aspects:

### Quality
Measures whether generated content is complete, legible, clear, and physically plausible.

### Aesthetics
Measures visual appeal, composition, color harmony, and stylistic coherence.

### Safety
Captures safety-related properties including harmful content, risky behavior, discrimination, intellectual property concerns, and the obviousness of generative artifacts.

---

## Visual Forms

VisForm spans **14 high-level categories**, including:

| Category | Examples |
|----------|----------|
| General Photography | Realistic photos, portraits |
| Specialized Photography | Macro, aerial, underwater |
| Traditional Painting | Oil painting, Chinese ink painting |
| Creative and Conceptual Art | Abstract art, surrealism |
| Illustration and Comics | Manga, storyboard |
| Crafts | Paper cutting, embroidery |
| Sculpture and Objects | 3D renders, product shots |
| Digital Graphics | Film posters, album covers |
| Scientific Imaging | CT images, microscopy |
| Diagrams | Flowcharts, architecture |
| Data Visualization | Charts, infographics |
| Sensor Data | Depth maps, thermal images |
| Patterns | Textures, seamless patterns |
| Design Elements | Icons, logos, UI elements |

---

## Use Cases

VisForm is intended for:

- Benchmarking generative image models
- Evaluating automatic image quality metrics
- Studying robustness under domain shift
- Analyzing expert judgments of generated images
- Comparing model families across visual forms
- Developing new evaluation metrics for quality, aesthetics, and safety

---

## Paper

**Evaluating Generative Models via One-Dimensional Code Distributions**
Zexi Jia, Pengcheng Luo, Yijia Zhong, Jinchao Zhang, Jie Zhou
**CVPR 2026**

arXiv: [2603.08064](https://arxiv.org/abs/2603.08064)

---

## Citation

If you use VisForm in your research, please cite:

```bibtex
@article{jia2026evaluating,
  title={Evaluating Generative Models via One-Dimensional Code Distributions},
  author={Jia, Zexi and Luo, Pengcheng and Zhong, Yijia and Zhang, Jinchao and Zhou, Jie},
  journal={arXiv preprint arXiv:2603.08064},
  year={2026}
}
```
