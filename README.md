# Human Pose Estimation Project

## Overview
This project focuses on research and implementation of bottom-up human pose estimation, with an emphasis on adaptive, scale-aware heatmap generation and integration with state-of-the-art frameworks. It combines literature review, code experiments, and workflow diagrams to support experimentation and reproducible research.

## Project Structure
- **simple_bottom_up_pose.py**: Main Python script for adaptive heatmap generation, visualization, and prototyping.
- **articles/**: Research articles, tables, and supporting materials organized by research objectives.
- **external_repos/**: Cloned external repositories (e.g., HRNet, HigherHRNet, SWAHR, Scale-sensitive-Heatmap) for reference and code adaptation.
- **Markdown Files**: 
  - `Human Pose Estimation Workflow.md`: Step-by-step workflow.
  - `Enhancing_Bottom_Up_Human_Pose_Estimation.md`: Pipeline improvements.
  - `Literature_Review.md`: Key research and findings.
  - `relevant_articles_comprehensive.md`: Curated article list.
  - `research_objectives_articles.md`: Research objectives and mapped articles.
- **.puml files**: PlantUML diagrams for visualizing frameworks and pipelines.

## Getting Started
1. **Review Documentation**: Start with the markdown files for workflow, literature review, and research objectives.
2. **Run Adaptive Heatmap Demo**:
   - Ensure Python 3.8+ is installed.
   - Install requirements: `pip install numpy matplotlib`
   - Run: `python simple_bottom_up_pose.py` to generate and visualize adaptive heatmaps.
3. **Explore External Repos**: Reference or adapt code from `external_repos/` for advanced experiments or integration with frameworks like HRNet or SWAHR.
4. **Visualize Diagrams**: Use PlantUML to render `.puml` files:
   ```bash
   java -jar plantuml.jar -tpng <diagram>.puml
   ```
5. **Browse Research**: See the `articles/` directory for grouped research, tables, and supporting documents by objective.

## Requirements
- Python 3.8+ (for .py scripts)
- numpy, matplotlib (for visualization)
- PlantUML (for diagram rendering)

## Contribution
Contributions are welcome! Add new research articles, code experiments, or update documentation to support ongoing research in human pose estimation.

---

For detailed workflows and research context, refer to the included markdown files and articles. For adaptive heatmap generation and visualization, see `simple_bottom_up_pose.py`.
