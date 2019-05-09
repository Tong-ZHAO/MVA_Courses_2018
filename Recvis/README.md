# Object recognition and computer vision

* **Professor**: Teached by [Ivan Lpatev](https://scholar.google.com/citations?user=-9ifK0cAAAAJ&hl=en), [Jean Ponce](https://scholar.google.com/citations?user=vC2vywcAAAAJ&hl=en), [Cordelia Schmid](https://scholar.google.com/citations?user=IvqCXP4AAAAJ&hl=en) and [Josef Sivic](https://scholar.google.com/citations?user=NCtKHnQAAAAJ&hl=en) (ENS Ulm).

* **Language**: The course is taught in English.

## Objective

Automated object recognition - and more generally scene analysis - from photographs and videos is the great challenge of computer vision. The objective of this course is to provide a coherent introductory overview of the image, object and scene models, as well as the methods and algorithms, used today to address this challenge.

## Validation

The course is validated by a project (50%) and 3 TPs (50%).

The project is evaluated by a proposal (10%) at the beginning, a presentation (30%) and a report (60%) at the end.

The TPs focus on the following algorithms:

* H1: Instance-level recognition (SIFT)
* H2: Neural networks for image classification
* H3: Image classification challenge on Kaggle ([website](https://www.kaggle.com/c/mva-recvis-2018))

## My Project

With Oscar C., we worked on the paper: [Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images](http://openaccess.thecvf.com/content_ECCV_2018/papers/Nanyang_Wang_Pixel2Mesh_Generating_3D_ECCV_2018_paper.pdf), an end-to-end deep learning architecture that produces a 3D shape in triangular mesh from a single color image. We implemented it in PyTorch and the code can be found [here](https://github.com/Tong-ZHAO/Pixel2Mesh-Pytorch). Based on the author's algorithm, we improved it by introducing a U-Net auto-encoder to reconstruct the image, which helps the net to converge faster.