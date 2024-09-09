import paradance as para


selected_columns = ['live_fr_living_ctr', 'live_fr_living_lvtr', 'live_fr_living_gtr', 'fr_pvtr', 'total_cost_amt', 'live_click']
formula = "targets[0]"  # define the formula for the objective function

loader = para.CSVLoader(
    file_path="./",
    file_name="samples",
    clean_zero_columns=['live_fr_living_ctr', 'live_fr_living_lvtr', 'live_fr_living_gtr', 'fr_pvtr', 'total_cost_amt',],
    max_rows=100000,
)

weights_num=2
equation_type='json'
equation_json = {
    "formula": {
        "live_ue_score#_no": "(1.0 + 64.345*live_fr_living_ctr)^0.1322",
        "live_gift_score#_no": "2.5*live_fr_living_gtr*weights[0]",
        "live_gift_score_ceil#_no": "live_gift_score",
        "final_score#_no": "weights[1]*live_ue_score + live_gift_score_ceil"
    }
}

cal = para.Calculator(
    df=loader.df,
    selected_columns=selected_columns,
    equation_type=equation_type,  # "sum" or "product" or "free_style or "json"
    equation_json=equation_json,
)

ob = para.MultipleObjective(
    calculator=cal,
    direction="maximize",
    formula=formula,
    weights_num=weights_num,
    study_name="Make KS Great Again",
    free_style_lower_bound=[0, -1,],
    free_style_upper_bound=[1000, 100,],
)

ob.add_evaluator(
    flag="tau",
    target_column="app_use_duration",
)

para.optimize_run(ob, n_trials=100)
