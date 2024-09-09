from paradance import LogarithmPCAPipeline

pipe = LogarithmPCAPipeline(
    config_path='config_lpca.yml',
    n_trials=200,
)

pipe.run()
pipe.show_results()
