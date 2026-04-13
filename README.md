# ChronicAI-platform

ChronicAI-platform is a Streamlit-based multitarget screening and prioritization framework for chronic-disease drug discovery. The platform profiles compounds across a Core-21 target panel, reports predicted pChEMBL-centered outputs, and supports developability-aware triage through transparent ranking views and exportable figures.

## Overview

ChronicAI-platform is designed for computational triage rather than experimental confirmation. The app supports:

- compound upload and preprocessing
- structure validation and deduplication
- descriptor calculation
- multitarget screening across 21 targets
- disease-group-aware prioritization
- target-by-compound heatmaps
- exportable manuscript-ready tables and figures

The current interface is organized around four pages:

1. **Overview**
2. **Input compounds**
3. **Multitarget screening**
4. **Prioritization dashboard**

## Core capabilities

- **Core-21 multitarget panel** spanning oncology, neurology, cardiovascular, metabolic, and diabetes-related targets
- **Predicted pChEMBL outputs** for deployment-oriented ranking
- **Developability-aware triage** using Lipinski, Veber, and Egan rule summaries
- **Flexible visualization** with multiple chart types and export options
- **Transparent prioritization** using maximum potency, mean potency, active-like breadth, QED, and developability support
- **Streamlit deployment-ready design** for interactive screening and review

## Target groups

The platform currently includes five disease-group categories:

- Oncology
- Neurology
- Cardiovascular
- Metabolic
- Diabetes-related

## Project structure

A clean project layout can follow this structure:

```text
ChronicAI-platform/
├── chronicai_app.py
├── requirements.txt
├── README.md
├── .gitignore
├── assets/
├── example_data/
└── outputs/