import paradance as para

selected_columns = ['live_fr_living_ctr', 'live_fr_living_lvtr', 'live_fr_living_gtr', 'fr_pvtr',]

formula = "-10 * targets[0] + targets[1] - targets[2]"  # define the formula for the objective function

loader = para.CSVLoader(
    file_path="./",
    file_name="samples",
    clean_zero_columns=selected_columns,
    max_rows=200000,
)

cal = para.Calculator(
    df=loader.df,
    selected_columns=selected_columns,
    equation_type="product",  # "sum" or "product" or "free_style or "json"
)

ob = para.MultipleObjective(
    calculator=cal,
    direction="maximize",
    formula=formula,
    study_name="Make KS Great Again",
    first_order_scale_lower_bound=1, # Limit the product of linear terms and features to a few orders of magnitude below 1e0.
    max_min_scale_ratio=3, # "Constrain the ratio of max to min for the product of linear terms and features"
)

ob.add_evaluator(
    flag="portfolio",
    target_column="live_click",
)

ob.add_evaluator(
    flag="auc",
    target_column="ua_count",
)

ob.add_evaluator(
    flag="distinct_count_portfolio",
    target_column="user_id",
)

para.optimize_run(ob, n_trials=100)
