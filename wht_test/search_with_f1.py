import paradance as para
import pandas as pd
import time
import os


def process_data(data_path, column_names, feature_cols, label_col, save_path, product_name=None):
    """ 处理原始数据，生成label，并导出为新的csv文件 """
    print(f'read data at {data_path}...')
    time0=time.time()
    df=pd.read_csv(data_path, sep='\t', header=None,names=column_names, na_values='\\N')
    print(f'read data done! {(time.time()-time0)/60:.1f} min')

    print('dataframe:')
    print(df.head())

    # drop NA
    print('dataframe length:', len(df))
    df.dropna(inplace=True)
    print('dataframe length after dropna:', len(df))

    if product_name:
        df = df[df['product_name'] == product_name]
    print('dataframe length after fintering product:', len(df))

    # type converting
    print('convert dtype...')
    time0=time.time()
    convert_dict = {k:float for k in feature_cols}
    convert_dict[label_col] = int
    df = df.astype(convert_dict)
    print(f'convert dtype done! {(time.time()-time0)/60:.1f} min')

    # generate new label
    print('generate new label and groupby column...')
    time0=time.time()
    df['new_label'] = df[label_col].apply(lambda x: 1 if x > 0 else 0)
    df['llsid_pid'] = df['llsid'].astype(str) + df['photo_id'].astype(str)
    # 移除只有一个label的组
    id_label_counts = df.groupby('llsid_pid')['new_label'].nunique()
    ids_to_keep = id_label_counts[id_label_counts > 1].index
    df = df[df['llsid_pid'].isin(ids_to_keep)]
    print(f'generate new label and groupby column done, {(time.time()-time0)/60:.1f} min')

    print('dataframe:')
    print(df.head())

    print(f'save csv to {save_path}')
    df.to_csv(save_path, index=False)
    print('save csv done!')


def main():
    ## 处理原始数据，导出为新的csv
    print('process data...')
    data_path='data/train/20240908'
    column_names = ['llsid', 'photo_id', 'label', 'norm_rerank_like_pxtr', 'norm_rerank_reply_pxtr', 'norm_rerank_expand_pxtr', 'product_name']
    feature_cols = ['norm_rerank_like_pxtr', 'norm_rerank_reply_pxtr', 'norm_rerank_expand_pxtr']   # 与 init_values_a 等参数顺序对应
    label_col = 'label'
    product_name='KUAISHOU'   # NEBULA | KUAISHOU
    group_by_cols = ['llsid', 'photo_id']
    save_path = f'data/train_{product_name}_{data_path.split('/')[-1]}.csv'
    if not os.path.exists(save_path):
        process_data(data_path,column_names,feature_cols,label_col,save_path=save_path,product_name=product_name)

    
    ## 加载数据
    print('load data...')
    loader = para.CSVLoader(
        file_path="./data/",
        file_name=f"train_{product_name}_{data_path.split('/')[-1]}",
        clean_zero_columns=['norm_rerank_like_pxtr', 'norm_rerank_reply_pxtr', 'norm_rerank_expand_pxtr'],
        # max_rows=100000,
    )
    print(loader.df.head())

    

    ## 配置公式形式
    print('setting equation and calculator...')
    selected_columns = ['norm_rerank_like_pxtr', 'norm_rerank_reply_pxtr', 'norm_rerank_expand_pxtr']

    weights_num=9
    equation_type='json'
    equation_json = {
        "formula": {
            "like_score": "(weights[0]+weights[1]*norm_rerank_like_pxtr)^(weights[2])",
            "reply_score": "(weights[3]+weights[4]*norm_rerank_reply_pxtr)^(weights[5])",
            "expand_score": "(weights[6]+weights[7]*norm_rerank_like_pxtr)^(weights[8])",
            "final_score": "like_score * reply_score * expand_score"
        }
    }

    cal = para.Calculator(
        df=loader.df,
        selected_columns=selected_columns,
        equation_type=equation_type,  # "sum" or "product" or "free_style or "json"
        equation_json=equation_json,
    )


    ## 定义优化目标
    print('setting objective...')
    formula = "targets[0]"  # define the formula for the objective function. targets[0]对应第一个evaluator的评估结果，formula是targets(list)的公式。formula的结果就是要优化的目标。
    ob = para.MultipleObjective(
        calculator=cal,
        direction="maximize",
        formula=formula,
        weights_num=weights_num,
        study_name="nebula-f1",
        # free_style_lower_bound=[0, -1,],  # 定义每个weight的上下界
        # free_style_upper_bound=[1000, 100,],
    )

    ob.add_evaluator(
        flag="wuauc",
        target_column="new_label",
        groupby='llsid_pid',
    )


    ## 计算 baseline 的结果
    print('calculate baseline...')
    base_weights=[0.3,3,2,0.3,3,2,0.3,5,3]
    cal.get_overall_score(base_weights) # 在计算离线指标前需要先算一次融合分
    base_result = cal.calculate_wuauc(
        groupby = 'llsid_pid',
        target_column = 'new_label',
        mask_column=None
    )
    print(f'baseline: wuauc={base_result}')


    ## 开始优化
    print('start optimizing...')
    para.optimize_run(ob, n_trials=100)
    print('optimization is done!')

if __name__=='__main__':
    main()