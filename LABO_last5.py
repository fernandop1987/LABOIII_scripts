# %%
# Cargar sell-in.txt (puede ser un archivo grande, leer solo columnas necesarias)
import pandas as pd
sellin_cols = ['periodo', 'customer_id', 'product_id', 'plan_precios_cuidados', 'cust_request_qty', 'cust_request_tn', 'tn']
df_sellin = pd.read_csv('sell-in.txt', sep='\t', usecols=sellin_cols)
df_sellin.head()

# %%
# Contar valores únicos de customer_id
df_sellin['customer_id'].nunique()
# Contar valores únicos de product_id
#df_sellin['product_id'].nunique()
# Contar valores únicos de periodo
#df_sellin['periodo'].nunique()


# %%
import pandas as pd

# Cargar product_id_apredecir201912.txt
df_ids = pd.read_csv('product_id_apredecir201912.txt')
df_ids.head()

# %%
# Merge de ambos dataframes por product_id
#df_merged = pd.merge(df_sellin, df_ids, on='product_id', how='inner')
#df_merged.head()

# %%
# Si 'periodo' es tipo string o int, conviértelo a datetime para mayor facilidad
df_sellin['periodo'] = pd.to_datetime(df_sellin['periodo'], format='%Y%m')
df_sellin= df_sellin.sort_values(['product_id', 'customer_id', 'periodo']).reset_index(drop=True)


# %%
# Leo archivo parquet
import pandas as pd
df_full = pd.read_parquet("df_full.parquet")

# %% [markdown]
# ---

# %% [markdown]
# #### Identificar clientes importantes

# %%
# Consistencia: cuántos meses tiene ventas
ventas_mensuales = df_full[df_full['tn'] > 0].groupby('customer_id')['periodo'].nunique()

# Volumen: total de toneladas vendidas
ventas_totales = df_full.groupby('customer_id')['tn'].sum()

# Unir criterios
clientes_importantes = ventas_mensuales[(ventas_mensuales >= 30)]\
    .index.intersection(ventas_totales[ventas_totales >= 10000].index)
print(clientes_importantes)
#clientes_importantes = set(clientes_importantes)


# %%
# Asignar grupo de cliente
df_full['grupo_cliente'] = df_full['customer_id'].apply(lambda x: 'top' if x in clientes_importantes else 'resto')

# %% [markdown]
# ---

# %% [markdown]
# ## Entrenamiento del modelo Light GBM

# %% [markdown]
# Preparar los datos de entrenamiento y test
# Entrenamiento: Usa todos los datos donde tn_t_plus_2 no es NaN y el período es menor a 201912 (para no usar datos del futuro).
# Test: Filtra las filas donde el período es 201912 (diciembre 2019), ya que para esas filas queremos predecir tn en 202002 (febrero 2020).

# %%
# Lista de features: incluye todas las columnas que empiezan con los prefijos de los features
feature_cols = [
    col for col in df_full.columns
    if (
        col.startswith('tn_') or
        col.startswith('cust_request_qty_lag_') or
        col.startswith('cust_request_qty_roll') or
        col.startswith('cust_request_tn_lag_') or
        col.startswith('cust_request_tn_roll') or
        col == 'plan_precios_cuidados'  or
        #col == 'product_id' or
        col in ['month', 'quarter', 'year', 'month_sin', 'month_cos']  
    )
]

# Quito la columna 	tn_t_plus_2 de las features
feature_cols = [col for col in feature_cols if col != 'tn_t_plus_2']

# %%
# Verifico los valores de product_id = 20001 y customer_id = 10063
# train[(train['product_id'] == 20001) & (train['customer_id'] == 10063) & (train['periodo'] > pd.to_datetime('2019-01-01'))]
#df_full[(df_full['product_id'] == 20064) & (df_full['customer_id'] == 10001) & (df_full['periodo'] > pd.to_datetime('2019-01-01'))]

# %% [markdown]
# #### Modelo jerárquico

# %%
magicos = [
    20001, 20002, 20003, 20004, 20005, 20006, 20007, 20008, 20009, 20010,
    20011, 20012, 20013, 20014, 20015, 20016, 20017, 20018, 20019, 20020,
    20021, 20022, 20023, 20024, 20025, 20026, 20027, 20028, 20029, 20030,
    20031, 20032, 20033, 20035, 20037, 20038, 20039, 20041, 20042, 20043,
    20044, 20045, 20046, 20047, 20049, 20050, 20051, 20052, 20053, 20054,
    20055, 20056, 20057, 20058, 20059, 20061, 20062, 20063, 20065, 20066,
    20067, 20068, 20069, 20070, 20071, 20072, 20073, 20074, 20075, 20076,
    20077, 20079, 20080, 20081, 20082, 20084, 20085, 20086, 20087, 20089,
    20090, 20091, 20092, 20093, 20094, 20095, 20096, 20097, 20099, 20100,
    20101, 20102, 20103, 20106, 20107, 20108, 20109, 20111, 20112, 20114,
    20116, 20117, 20118, 20119, 20120, 20121, 20122, 20123, 20124, 20125,
    20126, 20127, 20129, 20130, 20132, 20133, 20134, 20135, 20137, 20138,
    20139, 20140, 20142, 20143, 20144, 20145, 20146, 20148, 20150, 20151,
    20152, 20153, 20155, 20157, 20158, 20159, 20160, 20161, 20162, 20164
]


# %%
# Separar datasets
filtro_clientes_top = df_full["customer_id"].isin(clientes_importantes)
filtro_productos_top  = df_full["product_id"].isin(magicos) 

#Productos importantes y clientes importantes
df_full_A = df_full[filtro_clientes_top & filtro_productos_top].copy()

#Productos importantes y resto de clientes
df_full_B = df_full[~filtro_clientes_top & filtro_productos_top].copy()

#Productos no importantes
df_full_C = df_full[~filtro_productos_top].copy()



df_full_B_agg = (
    df_full_B
    .groupby(['product_id', 'periodo'], as_index=False)
    .agg({**{f: 'sum' for f in feature_cols}, 'tn_t_plus_2': 'sum', 'tn': 'sum'})
)

df_full_C_agg = (
    df_full_C
    .groupby(['product_id', 'periodo'], as_index=False)
    .agg({**{f: 'sum' for f in feature_cols}, 'tn_t_plus_2': 'sum', 'tn': 'sum'})
)

# %%
# Nuevos cortes para validación
cutoff_dates = ['2019-08-01', '2019-09-01', '2019-10-01']

valid_sets_top = []
valid_sets_int = []
valid_sets_resto = []

for cutoff in cutoff_dates:
    # Producto-cliente importantes
    train_top = df_full_A[(df_full_A['periodo'] < cutoff) & df_full_A['tn_t_plus_2'].notnull()]
    valid_top = df_full_A[(df_full_A['periodo'] == cutoff) & df_full_A['tn_t_plus_2'].notnull()]
    valid_sets_top.append((train_top, valid_top))

    # Producto importante, cliente no importante
    train_int = df_full_B_agg[(df_full_B_agg['periodo'] < cutoff) & df_full_B_agg['tn_t_plus_2'].notnull()]
    valid_int = df_full_B_agg[(df_full_B_agg['periodo'] == cutoff) & df_full_B_agg['tn_t_plus_2'].notnull()]
    valid_sets_int.append((train_int, valid_int))

    # Resto de productos (validación con promedio)
    valid_resto = df_full_C_agg[(df_full_C_agg['periodo'] == cutoff) & df_full_C_agg['tn_t_plus_2'].notnull()]
    valid_sets_resto.append(valid_resto)

# %% [markdown]
# #### Entrena el modelo LightGBM

# %%
# Optimización con Optuna del modelo jerárquico

import numpy as np
import lightgbm as lgb
import optuna

def run_optuna_study_multi_validation(
    train_sets,  # Lista de tuplas: [(X_train_1, y_train_1, w_train_1), ...]
    valid_sets,  # Lista de tuplas: [(X_val_1, y_val_1, w_val_1), ...]
    study_name
):
    def total_forecast_error(y_true, y_pred):
        error_abs = np.abs(y_true - y_pred)
        return np.sum(error_abs) / np.sum(y_true)

    def objective(trial):
        param = {
            'objective': 'regression',
            'metric': 'mse',
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'n_estimators': trial.suggest_int('n_estimators', 500, 1000),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'linear_tree': True
        }

        tfe_list = []

        for i, ((X_train, y_train, w_train), (X_val, y_val, w_val)) in enumerate(zip(train_sets, valid_sets)):
            model = lgb.LGBMRegressor(**param)
            model.fit(X_train, y_train, sample_weight=w_train)

            y_pred = model.predict(X_val)
            tfe = total_forecast_error(y_val, y_pred)

            tfe_list.append(tfe)
            trial.set_user_attr(f'tfe_period_{i+1}', tfe)

        trial.set_user_attr('tfe_list', tfe_list)
        return np.mean(tfe_list)

    storage = optuna.storages.RDBStorage(url="sqlite:///optuna_study.db")

    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',
        storage=storage,
        load_if_exists=True
    )

    study.optimize(objective, n_trials=100)

    print(f"Study '{study_name}' - Best params:", study.best_params)
    print(f"Study '{study_name}' - Best TFE promedio en {len(valid_sets)} períodos:", study.best_value)

    return study


# %%
# ---------- 1. Cliente-producto importantes ----------
train_sets_top = []
valid_sets_top_final = []

for train_df, valid_df in valid_sets_top:
    X_train = train_df[feature_cols]
    y_train = train_df['tn_t_plus_2']
    w_train = train_df['tn'].clip(lower=0.1)

    X_val = valid_df[feature_cols]
    y_val = valid_df['tn_t_plus_2']
    w_val = valid_df['tn'].clip(lower=0.1)

    train_sets_top.append((X_train, y_train, w_train))
    valid_sets_top_final.append((X_val, y_val, w_val))

study_top = run_optuna_study_multi_validation(
    train_sets=train_sets_top,
    valid_sets=valid_sets_top_final,
    study_name="tuning_top_multival3"
)


# %%
# ---------- 2. Producto importante, cliente no importante ----------
train_sets_int = []
valid_sets_int_final = []

for train_df, valid_df in valid_sets_int:
    X_train = train_df[feature_cols]
    y_train = train_df['tn_t_plus_2']
    w_train = train_df['tn'].clip(lower=0.1)

    X_val = valid_df[feature_cols]
    y_val = valid_df['tn_t_plus_2']
    w_val = valid_df['tn'].clip(lower=0.1)

    train_sets_int.append((X_train, y_train, w_train))
    valid_sets_int_final.append((X_val, y_val, w_val))

study_int = run_optuna_study_multi_validation(
    train_sets=train_sets_int,
    valid_sets=valid_sets_int_final,
    study_name="tuning_int_multival2"
)

# %%
import optuna.visualization as vis

# Visualizar historial de optimización
vis.plot_optimization_history(study_top).show()
vis.plot_optimization_history(study_int).show()


# %% [markdown]
# #### Entrenamiento del modelo jerárquico

# %%
import numpy as np
import lightgbm as lgb

# Obtener los mejores parámetros de cada estudio
best_params_top = study_top.best_params
best_params_int = study_int.best_params

# Agregar los parámetros fijos
best_params_top.update({
    'objective': 'regression',
    'metric': 'mse',
    'linear_tree': True
})

best_params_int.update({
    'objective': 'regression',
    'metric': 'mse',
    'linear_tree': True
})


# Productos y clientes importantes
# Unificamos train y valid 
train_top_final = df_full_A[(df_full_A['periodo'] <= '2019-12-01') & (df_full_A['tn_t_plus_2'].notnull())]

model_top = lgb.LGBMRegressor(**best_params_top)
model_top.fit(
    train_top_final[feature_cols],
    train_top_final['tn_t_plus_2'],
    sample_weight=train_top_final['tn'].clip(lower=0.1)
)


# Productos importantes y clientes no importantes
# Unificamos train y valid para productos importantes
train_int_final = df_full_B_agg[(df_full_B_agg['periodo'] <= '2019-12-01') & (df_full_B_agg['tn_t_plus_2'].notnull())]

model_int = lgb.LGBMRegressor(**best_params_int)
model_int.fit(
    train_int_final[feature_cols],
    train_int_final['tn_t_plus_2'],
    sample_weight=train_int_final['tn'].clip(lower=0.1)
)




# %% [markdown]
# ---

# %% [markdown]
# #### Predicciones para febrero 2020

# %%
# Predicciones para df_full_A (modelo a nivel producto-cliente)

df_pred_cliente = df_full_A[df_full_A['periodo'] == '2019-12-01'].copy()
df_pred_cliente['pred'] = model_top.predict(df_pred_cliente[feature_cols])
df_pred_cliente_grouped = df_pred_cliente.groupby('product_id')['pred'].sum().reset_index()
df_pred_cliente_grouped.head()

# %%
# Predicciones para df_full_B_agg (modelo a nivel producto)

df_pred_int = df_full_B_agg[df_full_B_agg['periodo'] == '2019-08-01'].copy()
df_pred_int['pred'] = model_int.predict(df_pred_int[feature_cols])
df_pred_int_grouped = df_pred_int.groupby('product_id')['pred'].sum().reset_index()
df_pred_int_grouped.head()  

# %%
# Alternativa: Predicciones para df_full_C con promedio de últimos 12 meses
# Filtrar los últimos 12 meses
ultimo_mes_ = '2019-12-01'
primer_mes_ = '2019-01-01'

df_avg = df_full_C_agg[
    (df_full_C_agg['periodo'] >= primer_mes_) &
    (df_full_C_agg['periodo'] <= ultimo_mes_)
]

# Calcular promedio de tn por producto
tn_avg_by_product = df_avg.groupby('product_id')['tn'].mean().reset_index()
tn_avg_by_product.rename(columns={'tn': 'pred'}, inplace=True)

# Usar ese promedio como predicción para 2019-12-01
df_pred_prod_groupedA = tn_avg_by_product.copy()
df_pred_prod_groupedA.head()


# %%
# Unir las tres predicciones

df_pred_final = pd.concat([df_pred_cliente_grouped, df_pred_prod_groupedA, df_pred_int_grouped], axis=0) 

# Volver a agrupar para sumar predicciones
df_pred_final = df_pred_final.groupby('product_id', as_index=False)['pred'].sum()
df_pred_final.rename(columns={'pred': 'tn'}, inplace=True)
df_pred_final.head()

# %% [markdown]
# #### Archivo para Kaggle

# %%
# Archivo a partir de modelo jerárquico

# Merge con dataframe ids por product_id
submission_mj = pd.merge(df_pred_final, df_ids, on='product_id', how='inner')
#submission_mj.head()
#submission_mj.shape

# Filtrar valores negativos de tn y cambiar a 0
submission_mj.loc[submission_mj['tn'] < 0, 'tn'] = 0


# Exportar a CSV 
submission_mj.to_csv('submission_mj.csv', index=False)



