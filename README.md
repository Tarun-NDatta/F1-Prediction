[comment]: # (You may find the following markdown cheat sheet useful: https://www.markdownguide.org/cheat-sheet/. You may also consider using an online Markdown editor such as StackEdit or makeareadme.) 

## Project title: *AI/ML-Based Formula 1 Race Outcome Prediction Using Historical and Real-Time Data*

### Student name: *Tarun Datta*

### Student email: *td188@student.le.ac.uk*

### Project description: 
This project leverages AI/ML techniques to predict Formula 1 race outcomes, including qualifying and final race positions. It uses historical data, driver profiles, team strategies, and environmental variables to extract patterns and trends in performance. The end goal is to develop a predictive model and an interactive web dashboard to visualize results and provide insights. This real-world data-driven project supports learning in machine learning, cloud computing, and data engineering. It showcases the application of theory to a dynamic, strategy-intensive sport.

### List of requirements (objectives): 

Essential:
- Collect historical F1 race data from the 2022 season onward via APIs.
- Preprocess and store structured race data for model training.
- Engineer meaningful features from driver, team, track, and race metrics.
- Build and evaluate ML models to predict qualifying and race outcomes.
- Develop a Django-based interface to visualize predictions and analysis.

Desirable:
- Analyze feature importance to understand what factors impact race results most.
- Implement model comparison (Random Forest, XGBoost, Neural Networks).
- Include qualifying data, weather impact, and team performance metrics in models.
- Allow dynamic switching between seasons (2022, 2023, etc.)

Optional:
- Enable real-time data ingestion for live race predictions.
- Integrate cloud services (e.g., AWS/GCP) for model hosting and scalability.
- Provide an option for users to compare model predictions vs actual race results.

## Information about this repository
This is the repository used individually for developing the dissertation project. It includes all software artefacts such as:

- `data/`: For data gathering, cleaning, and preprocessing scripts (via Django management commands).
- `prediction/`: For storing model-related files, training scripts, and evaluation logic.
- `dashboard/`: For Django views, templates, and routes to display results.
- `notebooks/`: For prototyping and exploratory data analysis using Jupyter Notebooks.
- `README.md`: This file.

Frequent commits are encouraged after each working feature or milestone. Commit messages should be clear and concise (e.g., "Added feature engineering for driver consistency", "Connected XGBoost model to prediction pipeline").
