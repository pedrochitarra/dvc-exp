quality_checks:
	echo "Running quality checks"
	isort .

pipeline_dag:
	echo "Visualizing pipeline"
	dvc dag --md > dag.md

build_model: quality_checks pipeline_dag
	echo "Building model"
	dvc repro

create_plots: build_model
	echo "Creating plots"
	dvc plots show

dvc_push: create_plots
	echo "Pushing model and data to remote"
	dvc push