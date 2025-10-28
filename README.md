# COWNET-AI-MultiAgent
Multi-agent framework for translating social network analytics into explainable dairy management decisions - part of the MooAnalytica Digital Livestock AI Suite

CowNet-AI: Multi-Agent Framework for Translating Social Network Analytics into Explainable Dairy Management Decisions
Overview
CowNet-AI is an open-source, modular, multi-agent framework developed by the MooAnalytica Research Group at Dalhousie University.
It integrates social network analysis (SNA), AI-driven behavioral modeling, and explainable decision support to transform raw sensor data (UWB tracking, accelerometers, video) into actionable insights for precision dairy management.

Key Features
Automated Social Network Analysis — Computes dynamic herd metrics (centrality, modularity, density) using trajectory and interaction data.
Multi-Agent Architecture — Distributed agents for data ingestion, SNA computation, welfare assessment, and decision reasoning.
Explainable AI Layer — Generates interpretable welfare and management recommendations via symbolic-AI reasoning.
Sensor Integration — Compatible with UWB tracking, accelerometer, and environmental sensors.
Open API and Visualization Dashboards — For farm-scale analytics and cross-system integration.

CowNet AI Multi-agent
+-------------------------------------------------------------+
|                     CowNet-AI Framework                     |
|-------------------------------------------------------------|
|  Data Agent  |  SNA Agent  |  Behavior Agent  |  Decision Agent |
|-------------------------------------------------------------|
|         Knowledge Graph + Explainability Layer              |
+-------------------------------------------------------------+
|             Visualization + API Interface                   |
+-------------------------------------------------------------+

Example Applications

Social hierarchy mapping and group dynamics tracking
Detection of social isolation and welfare anomalies
Predictive modeling of feeding and rumination networks
Farm-level decision dashboards for herd management

Installation

git clone https://github.com/mooanalytica/COWNET-AI-MultiAgent.git
cd COWNET-AI-MultiAgent
pip install -r requirements.txt

--
Usage Example
from cownet_ai import CowNet
model = CowNet(data_source='uwb_tracking_data.csv')
model.run_network_analysis()
model.visualize_network()
---
Contributors

Dr. Suresh Raja Neethirajan – Principal Investigator
Tahseen Shanteer - Research Intern
MooAnalytica Research Group, Dalhousie University
--
License
Distributed under the MIT License. See LICENSE for details. 
---
