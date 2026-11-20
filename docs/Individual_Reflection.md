# Individual Reflection: DS-01 and DS-02

**Name:** Aadarsh Kainthola  |  **ID:** NEX-INT-2026-3390

## What I owned
I worked on both projects on my own, from data to deployment.

- **DS-01 (Churn and CLV):** loaded the Telco data into PostgreSQL, did EDA, trained a LightGBM churn model (test AUC 0.829) and explained it with SHAP, ran Kaplan-Meier and Cox survival analysis, built an uplift model on the Hillstrom campaign data, calculated customer lifetime value, segmented customers with KMeans, and built and deployed a Streamlit ROI simulator.
- **DS-02 (Demand Forecasting):** built lag, rolling and holiday features, compared naive, Prophet, LightGBM and a Temporal Fusion Transformer, produced 80% prediction intervals, reconciled store and total forecasts, wrote an OR-Tools model for order quantities, and deployed a Plotly Dash app.
- I also wrote the READMEs, set up the GitHub repository and handled both deployments.

## What I learned
- A model that predicts churn well does not tell you who will respond to an offer. Uplift modeling answers that, and my results showed some customers with negative uplift, so contacting them can backfire.
- Survival analysis gave me "how long" instead of only "will they leave". Contract type was the strongest factor in the Cox model.
- A simple, well-built baseline is hard to beat. LightGBM with lag features reduced WAPE from 0.166 to 0.113, while my Temporal Fusion Transformer (0.353) did worse because it was small, trained on CPU for few epochs, and forecast 30 days ahead instead of 7. I learned to report this honestly instead of hiding it.
- Reconciliation makes forecasts consistent: the gap between store forecasts and the total went from 241.5 to about zero.
- Deployment problems are real work. I fixed a database driver mismatch, a wrong file path on Streamlit Cloud, and a missing PyTorch dependency in Darts that needed a session restart.

## Challenges
- Some libraries (CausalML) did not install in the newer Python environment, so I implemented the T-learner directly with LightGBM.
- The forecasting dataset has no price or promotion columns, so those features from the brief could not be built.
- Limited free compute made it hard to train the transformer properly.

## What I would do next
Use a dataset with price and promotions (such as M5), tune and train the transformer on a GPU, add more hierarchy levels, and replace the assumed costs in the optimizer and ROI simulator with real business numbers.

## Use of AI assistance
I used an AI assistant (Claude) for guidance and code help. I ran and debugged the code myself, and the results and interpretations above come from my own runs.
