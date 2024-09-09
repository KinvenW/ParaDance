from paradance import ClassicalPipeline

pipe = ClassicalPipeline(
    config_path='config_classical.yml',
    n_trials=200,
)

pipe.run()
pipe.show_results()
