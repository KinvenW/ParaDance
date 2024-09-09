import paradance as para

selected_columns = ['live_fr_living_ctr', 'live_fr_living_lvtr',]
formula = "-10 * targets[0] + targets[1] - targets[2]"  # define the formula for the objective function

equation_type='free_style'
equation_eval_str = '1 + weights[0] * columns[0] + weights[1] * columns[1] + weights[2] * columns[1]'
weights_num=3

loader = para.CSVLoader(
    file_path="./",
    file_name="samples",
    clean_zero_columns=selected_columns,
    max_rows=200000,
)

cal = para.Calculator(
    df=loader.df,
    selected_columns=selected_columns,
    equation_type=equation_type, # "sum" or "product" or "free_style or "json"
    equation_eval_str=equation_eval_str,
)

ob = para.MultipleObjective(
    calculator=cal,
    direction="maximize",
    formula=formula,
    weights_num=weights_num,
    free_style_lower_bound=[0, -1, 1],
    free_style_upper_bound=[1, 10, 100],
    study_name="Make KS Great Again",
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
