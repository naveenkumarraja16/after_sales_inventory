import subprocess

# Call analysis.py
subprocess.run(["python", "analysis.py"])

# Call analysisWithRegion.py
subprocess.run(["python", "analysisWithRegion.py"])

# Call EOQ_analysis.py
subprocess.run(["python", "EOQ_analysis.py"])

# Call newCodePredictionWithAnalysis.py
subprocess.run(["python", "newCodePredictionWithAnalysis.py"])

# Call newPredictionAnalysis.py
subprocess.run(["python", "newPredictionAnalysis.py"])

# Call dbscanNewAnalysisPredictionBasedOnProductName.py
subprocess.run(["python", "dbscanNewAnalysisPredictionBasedOnProductName.py"])

# Call newClusterPredictionBasedOnProductName.py
subprocess.run(["python", "newClusterPredictionBasedOnProductName.py"])
