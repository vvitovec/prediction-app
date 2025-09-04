# =============================================================================
# XGBOOST PREDIKTIVNÍ MODEL PRO CENY NEMOVITOSTÍ - UPRAVENÁ VERZE
# =============================================================================

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error, 
                            mean_absolute_percentage_error, median_absolute_error,
                            explained_variance_score, max_error)
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path='model_ready_data.csv'):

    print("Načítání dat...")
    df = pd.read_csv(file_path)
    
    df_clean = df.dropna(subset=['cena']).copy()
    print(f"Data po čištění: {df_clean.shape}")
    
    target = 'cena'
    features = [col for col in df_clean.columns if col != target and col != 'cena_za_m2']
    
    X = df_clean[features].copy()
    y = df_clean[target].copy()
    
    for col in X.columns:
        if X[col].isnull().any():
            if X[col].dtype in ['int64', 'float64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 0)
    
    print(f"Features: {len(features)}")
    print(f"Vzorky: {len(X)}")
    
    return X, y, features

def create_xgboost_model(X, y):

    print("\n=== KONFIGURACE A TRÉNOVÁNÍ MODELU ===")
    
    # Parametry modelu
    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'verbosity': 0
    }
    
    print(f"Training set: {X.shape[0]} vzorků")
    print(f"Počet features: {X.shape[1]}")
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X, y)
    
    y_pred = model.predict(X)
    
    metrics = calculate_detailed_metrics(y, y_pred)
    
    print("\n=== DETAILNÍ VÝSLEDKY MODELU ===")
    print(f"RMSE (Root Mean Squared Error): {metrics['rmse']:,.0f} Kč")
    print(f"MAE (Mean Absolute Error): {metrics['mae']:,.0f} Kč")
    print(f"MAPE (Mean Absolute Percentage Error): {metrics['mape']:.2f}%")
    print(f"MedAE (Median Absolute Error): {metrics['medae']:,.0f} Kč")
    print(f"R² (Coefficient of Determination): {metrics['r2']:.4f}")
    print(f"Explained Variance Score: {metrics['explained_variance']:.4f}")
    print(f"Max Error: {metrics['max_error']:,.0f} Kč")
    print(f"Mean Error: {metrics['mean_error']:,.0f} Kč")
    print(f"Standard Deviation of Errors: {metrics['std_error']:,.0f} Kč")
    
    if len(X) > 5:
        cv_scores = cross_val_score(model, X, y, cv=min(5, len(X)), 
                                   scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse = np.sqrt(-cv_scores)
        print(f"\nCross-Validation RMSE: {cv_rmse.mean():,.0f} ± {cv_rmse.std():,.0f} Kč")
        print(f"CV RMSE Range: {cv_rmse.min():,.0f} - {cv_rmse.max():,.0f} Kč")
    
    residuals = y - y_pred
    print(f"\n=== ANALÝZA REZIDUÍ ===")
    print(f"Mean Residual: {residuals.mean():,.0f} Kč")
    print(f"Median Residual: {residuals.median():,.0f} Kč")
    print(f"Residuals Std: {residuals.std():,.0f} Kč")
    print(f"Min Residual: {residuals.min():,.0f} Kč")
    print(f"Max Residual: {residuals.max():,.0f} Kč")
    
    error_percentiles = np.percentile(np.abs(residuals), [25, 50, 75, 90, 95])
    print(f"\n=== PERCENTILY ABSOLUTNÍCH CHYB ===")
    print(f"25th percentile: {error_percentiles[0]:,.0f} Kč")
    print(f"50th percentile (Median): {error_percentiles[1]:,.0f} Kč")
    print(f"75th percentile: {error_percentiles[2]:,.0f} Kč")
    print(f"90th percentile: {error_percentiles[3]:,.0f} Kč")
    print(f"95th percentile: {error_percentiles[4]:,.0f} Kč")
    
    return model, (X, y, y_pred, metrics, residuals)

def calculate_detailed_metrics(y_true, y_pred):

    metrics = {}
    
    metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred) * 100
    metrics['medae'] = median_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    metrics['explained_variance'] = explained_variance_score(y_true, y_pred)
    metrics['max_error'] = max_error(y_true, y_pred)
    
    errors = y_true - y_pred
    metrics['mean_error'] = np.mean(errors)
    metrics['std_error'] = np.std(errors)
    
    return metrics

def analyze_feature_importance(model, feature_names, top_n=15):

    print("\n=== DŮLEŽITOST FEATURES ===")
    
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"Top {top_n} nejdůležitějších features:")
    for i, (_, row) in enumerate(importance_df.head(top_n).iterrows()):
        print(f"{i+1:2d}. {row['feature']:<25} {row['importance']:.4f}")
    
    importance_df['cumulative_importance'] = importance_df['importance'].cumsum()
    features_80_percent = len(importance_df[importance_df['cumulative_importance'] <= 0.8])
    print(f"\nPočet features pokrývajících 80% důležitosti: {features_80_percent}")
    
    return importance_df

def predict_new_property(model, feature_names, **property_features):

    print("\n=== PREDIKCE NOVÉ NEMOVITOSTI ===")
    
    features = []
    print("Zadané vlastnosti:")
    for feature in feature_names:
        value = property_features.get(feature, 0)
        features.append(value)
        if value != 0:
            print(f"  {feature}: {value}")
    
    X_new = np.array(features).reshape(1, -1)
    predicted_price = model.predict(X_new)[0]
    
    print(f"\nPredikovaná cena: {predicted_price:,.0f} Kč")
    if 'rozloha_užitná' in property_features and property_features['rozloha_užitná'] > 0:
        print(f"Predikovaná cena za m²: {predicted_price/property_features['rozloha_užitná']:,.0f} Kč/m²")
    
    return predicted_price

def save_model(model, feature_importance_df, metrics, model_name='xgboost_real_estate_model'):

    print(f"\n=== UKLÁDÁNÍ MODELU ===")
    
    model_file = f'{model_name}.pkl'
    joblib.dump(model, model_file)
    print(f"Model uložen: {model_file}")
    
    importance_file = f'{model_name}_feature_importance.csv'
    feature_importance_df.to_csv(importance_file, index=False)
    print(f"Feature importance uloženo: {importance_file}")
    
    metrics_file = f'{model_name}_metrics.csv'
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metriky uloženy: {metrics_file}")
    
    return model_file, importance_file, metrics_file

def load_model(model_file):

    return joblib.load(model_file)
    
MODEL_PATH = 'xgboost_real_estate_model.pkl'
FEATURES = ['rozloha_užitná', 'počet_koupelen', 'počet_pokojů', 'garáž', 'balkón', 'balkón_plocha', 'lodžie', 'lodžie_plocha', 'mhd_dostupnost', 'parkování', 'sklep', 'sklep_plocha', 'terasa', 'terasa_plocha', 'výtah', 'dispozice_numeric', 'energetická_třída_numeric', 'město_encoded', 'typ_nemovitosti_encoded']

def load_model():
    return joblib.load(MODEL_PATH)

def predict_from_dict(model, features_dict):
    x = np.array([features_dict.get(f, 0) for f in FEATURES]).reshape(1, -1)
    price = model.predict(x)[0]
    return price
    
# =============================================================================
# HLAVNÍ SPUŠTĚNÍ
# =============================================================================

if __name__ == "__main__":
    print("XGBOOST MODEL PRO PREDIKCI CEN NEMOVITOSTÍ")
    print("=" * 70)
    
    X, y, features = load_and_prepare_data('model_ready_data.csv')
    
    model, results = create_xgboost_model(X, y)
    X_train, y_train, y_pred, metrics, residuals = results
    
    importance_df = analyze_feature_importance(model, features)
    
    model_file, importance_file, metrics_file = save_model(model, importance_df, metrics)
    
    print("\n" + "=" * 70)
    print("PŘÍKLAD POUŽITÍ:")
    
    example_property = {
        'rozloha_užitná': 85,
        'počet_koupelen': 1,
        'počet_pokojů': 3,
        'garáž': 1,  # True
        'balkón': 1,  # True
        'balkón_plocha': 6.0,
        'parkování': 1,  # True
        'výtah': 1,  # True
        'dispozice_numeric': 3.5,  # 3+kk
        'energetická_třída_numeric': 2,  # B
        'město_encoded': 184,  # Praha
        'typ_nemovitosti_encoded': 0  # Byt
    }
    
    predicted_price = predict_new_property(model, features, **example_property)
    
    print("\n" + "=" * 70)
    print("MODEL ÚSPĚŠNĚ VYTVOŘEN A ULOŽEN!")
    print(f"Pro nové predikce použijte: model = load_model('{model_file}')")
    print("Všechny metriky a výsledky byly uloženy do CSV souborů.")