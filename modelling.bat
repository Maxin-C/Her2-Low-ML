@echo off
python ./machine_learning.py ^
 --file_path "./dataset.xlsx" ^
 --feature_name "Her2 status" "Age" "Intravascular Cancer Thrombus" "Ki67" "Histologic type" "Stage" "Radiation Therapy" "Her2 target therapy" ^
 --mode "hr_pos" ^
 --os_tag "OS (5 years)" ^
 --dfs_tag "DFS (5 years)" ^
 --output_dir "./output"