# Artifact Appendix

Paper title: **SoK: (Un)usable Privacy: the Lack of Overlap between Privacy-Aware Sensing and Usable Privacy Research**

Artifacts HotCRP Id: **#8** 

Requested Badge:  **Available**, **Functional**, and **Reproduced**

## Description
This artifact repo contains 5 files: README.md (this file), `figure1.ipynb`, `figure1.py`, `papers_reference`, and `codebook_reference`. The artifact includes the sourcecode and datasets to generate Figure 1. `figure1.ipynb` and `figure1.py` contain the same code to generate Figure 1 in the paper in Jupyter Notebook and py-file formats, respectively. This code loads two datasets, one dataset consisting of 10,122 works' embeddings collected as part of this work (`papers_reference`) and another dataset consisting of the works included in the codebook and their embeddings (`codebook_reference`). 

Based on these datasets, a t-SNE plot is generated and the works are grouped along four academic categories: Human-Computer Interaction, Mobile Systems, Usability in Privacy, and Security and Privacy. An ellipse is fit for visualization purposes. On this plot, the works from the codebook are overlaid with gray crosses and orange diamonds representing the Privacy-Aware Sensing and Usable Privacy works found in the codebook, respectively. The code also computes the BC distance signifying the overlap for each pairwise combination of the four communities.

### Security/Privacy Issues and Ethical Concerns (All badges)
There are no security or privacy issues to the reviewer's machine. There are no ethical concerns regarding the artifacts.

## Basic Requirements (Only for Functional and Reproduced badges)

### Hardware Requirements
No specific hardware is required.

### Software Requirements
We use the Mamba Python environment with the anaconda package installed from the anaconda channel (which installs all the required packages). Additionally, `scikit-learn=1.2` is required. The Python version is 3.9.19. Directions to set up the environment can be found below. For those who do not wish to use Mamba, a `requirements.txt` file is included (but Python 3.9 is still required). 

Note: when using the `requirements.txt` file with `mamba install --file requirements.txt`, the correct conda channels must be set in the mamba environment (otherwise some packages in the correct version may not be found). Alternatively, pip can be used with `pip install -r requirements.txt`.

The datasets required for the artifact are in the `sok_artifact` folder along with the required py file or Jupyter notebook.

### Estimated Time and Storage Consumption
The artifact requires 113 seconds to process on a Standard VM and the total filesize is less than 200MB.

## Environment 

### Accessibility (All badges)
All artifact content can be found at https://github.com/ISC-Lab/sok_artifact 

The tag-id will be updated to the latest version once artifact has been evaluated.

### Set up the environment (Only for Functional and Reproduced badges)

To set up environment from scratch, Mamba, Anaconda standard packages, and `scikit-learn=1.2` are required.

To install Mamba:
```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```
Create sok venv,  install anaconda packages, and install `scikit-learn=1.2`:
```
mamba create -n sok python=3.9
mamba activate sok
mamba install -c anaconda anaconda
mamba install scikit-learn=1.2
```

### Testing the Environment (Only for Functional and Reproduced badges)

In the `sok_artifact` folder, run `python envtest.py` to verify all required packages and dataframes can be succesfully loaded.

Note to reviewers: For evaluation, a Standard VM (ID 173) has been set up with the instructions above.

### Using the Jupyter notebook
To use Jupyter notebooks, run `jupyter notebook` in the terminal when in the `sok_artifact` folder. A webserver will start and open a browser window to the Jupyter interface. Here, the `figure1.ipynb` notebook file can be opened. Alternatively, VSCode can be used with the Jupyter extension to open the `figure1.ipynb` file.

## Artifact Evaluation (Only for Functional and Reproduced badges)

### Main Results and Claims

#### Main Result 1: Lack of Overlap between Privacy-Aware Sensing and Usable Privacy Research

The works selected for inclusion in the codebook consist of Privacy-Aware Sensing (PAS) and Usable Privacy (UP) works. Qualitatively, in the SoK, we detailed the lack of overlap between these two research areas in their approaches, methodologies, and manner in which they evaluate their contributions. 

We additionally show, quantiatively using their embeddings, a lack of overlap in these two research domains. These collective works form three distinct clusters: one of UP works within the Usability in Privacy research community, another of PAS works within the Mobile Systems research community, and a small mixed cluster of UP and PAS works at the intersection of all four research communities that denotes emerging usable privacy-aware sensing research. This clustering, and their lack of overlap, is visualized in Figure 1.

#### Main Result 2: Direct Embeddings Analysis Shows Distinct Divide in How Contributions Talk About Usable Privacy-Aware Sensing

Given the lack of overlap between UP and PAS works, a direct embeddings analysis across the 4 stakeholder research communities (Human-Computer Interaction, Mobile Systems, Usability in Privacy, and Security and Privacy) offers additional evidence that individual communities have distinct “cultures" that influence how privacy mitigations for sensors are ideated, designed, and evaluated. Furthermore, the minimal overlap between certain combinations of communities further suggests a lack of cross-pollination, highlighted by the low density of works at the center of the t-SNE visualization, which may partially influence the minimal overlap in the methodologies for PAS and UP works. 

The lack of overlap is between the four stakeholder communites is visualized in Figure 1 and quantified in Table 5.

### Experiments 
List each experiment the reviewer has to execute. Describe:
 - How to execute it in detailed steps.
 - What the expected result is.
 - How long it takes and how much space it consumes on disk. (approximately)
 - Which claim and results does it support, and how.

#### Experiment 1: 10,122-paper t-SNE and BC Coefficients
This experiment will generate a t-SNE plot from the embeddings generated from 10,122 works collected across 12 academic venues. These works were grouped into 4 academic communities and the pairwise overlap in their distributions is computed.

`cd` to the `sok_artifact` folder, run `mamba activate sok` to activate the `sok` venv, run `python figure1.py`, which will generate `figure1.png` and compute and print the pairwise BC coefficients.

The generated `figure1.png` will match Figure 1 and the printed pairwise BC coefficients will match Table 5 in the paper. In Figure 1, the UP and PAS works form three distinct clusters as detailed in Main Result 1. Additionally, the individual research communties have distinct distributions, with lower density at the center of the visualization, showing a lack of overlap among the 4 stakeholder communities, as detailed in Main Result 2. Additionally, the BC values printed show lower overlap between specific pairs (e.g., HCI and S&P), which provides further evidence for a lack of overlap.

The experiment requires 113 seconds to process on a Standard VM and the total filesize is less than 200MB. Alternatively, you can use the Jupyter notebook `figure1.ipynb` to further interact with the dataset.

## Limitations (Only for Functional and Reproduced badges)
Tables 1-4 are subsets of or are derived from the codebook (Table 6 of the appendix) which is included in this artifact as `codebook_reference`. Table 5's BC values are computed in `figure1.py` and `figure1.ipynb`.

## Notes on Reusability (Only for Functional and Reproduced badges)
The 10,122 paper embeddings in `papers_reference` is a dataset contribution that can be reused for future SoK investigations to discover research gaps. Each entry has the work's title and  embedding with its URL included as a unique identifier. The Jupyter notebook `figure1.ipynb` is provided to further interact with the dataset.
