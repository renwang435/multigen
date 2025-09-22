# RoboVerse Documentation

1. Install the dependencies

```bash
conda create -n roboverse_page python=3.11
conda activate roboverse_page
pip install -r requirements.txt
```

2. Build the documentation and watch the change lively

```bash
rm -rf build/; make html; sphinx-autobuild ./source ./build/html
```
