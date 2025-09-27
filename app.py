import streamlit as st
import joblib
import numpy as np

def dispozice_to_features(dispozice_str):

    dispozice_map = {
        '1+kk': 1.0,
        '1+1': 1.5, 
        '2+kk': 2.0,
        '2+1': 2.5,
        '3+kk': 3.0,
        '3+1': 3.5,
        '4+kk': 4.0,
        '4+1': 4.5,
        '5+kk': 5.0,
        '5+1': 5.5,
    }
    

    pocet_pokoju = 3
    pocet_koupelen = 1
    dispozice_numeric = 3.5
    
    if dispozice_str in dispozice_map:
        dispozice_numeric = dispozice_map[dispozice_str]

        parts = dispozice_str.split('+')
        try:
            pocet_pokoju = int(parts[0])
        except:
            pocet_pokoju = 3
            

        druhy = parts[1].lower()
        if druhy == 'kk':
            pocet_koupelen = 1
        else:
            try:
                pocet_koupelen = int(druhy)
            except:
                pocet_koupelen = 1
    else:

        pocet_pokoju = 3
        pocet_koupelen = 1
        dispozice_numeric = 3.5
    
    return pocet_pokoju, pocet_koupelen, dispozice_numeric

def energeticka_trida_to_numeric(trida_str):

    trida_str = trida_str.upper().strip()
    map_tridy = {
        'A': 1,
        'B': 2, 
        'C': 3,
        'D': 4,
        'E': 5,
        'F': 6,
        'G': 7,
    }
    return map_tridy.get(trida_str, 3)
    
def load_city_map():
    city_map = {}
    with open('city_code_list.txt', encoding='utf-8') as f:
        for line in f:
            if ':' in line:
                name, code = line.strip().split(':', 1)
                city_map[name.strip()] = int(code.strip())
    return city_map

CITY_MAP = load_city_map()   

def preprocess_user_inputs(
    rozloha_uzitna,
    dispozice,
    energeticka_trida,
    balkon,
    balkon_plocha,
    lodzie,
    lodzie_plocha,
    sklep,
    sklep_plocha,
    terasa,
    terasa_plocha,
    garaz,
    mhd_dostupnost,
    parkovani,
    vytah,
    mesto_encoded,
    typ_nemovitosti_str,
):

    pocet_pokoju, pocet_koupelen, dispozice_numeric = dispozice_to_features(dispozice)
    
    energeticka_třída_numeric = energeticka_trida_to_numeric(energeticka_trida)
    
    balkon_val = 1 if balkon else 0
    lodzie_val = 1 if lodzie else 0
    sklep_val = 1 if sklep else 0
    terasa_val = 1 if terasa else 0
    garaz_val = 1 if garaz else 0
    vytah_val = 1 if vytah else 0
    
    balkon_plocha_val = balkon_plocha if balkon_val == 1 else 0
    lodzie_plocha_val = lodzie_plocha if lodzie_val == 1 else 0
    sklep_plocha_val = sklep_plocha if sklep_val == 1 else 0
    terasa_plocha_val = terasa_plocha if terasa_val == 1 else 0
    
    mhd_dostupnost_val = int(round(mhd_dostupnost))
    parkovani_val = int(round(parkovani))
    mesto_encoded_val = int(round(mesto_encoded))
    
    typ_nemovitosti_map = {'Byt': 0, 'Dům': 1}
    typ_nemovitosti_encoded = typ_nemovitosti_map.get(typ_nemovitosti_str, 0) 
   
    features_dict = {
        'rozloha_užitná': rozloha_uzitna,
        'počet_pokojů': pocet_pokoju,
        'počet_koupelen': pocet_koupelen,
        'dispozice_numeric': dispozice_numeric,
        'energetická_třída_numeric': energeticka_třída_numeric,
        'balkón': balkon_val,
        'balkón_plocha': balkon_plocha_val,
        'lodžie': lodzie_val,
        'lodžie_plocha': lodzie_plocha_val,
        'sklep': sklep_val,
        'sklep_plocha': sklep_plocha_val,
        'terasa': terasa_val,
        'terasa_plocha': terasa_plocha_val,
        'garáž': garaz_val,
        'mhd_dostupnost': mhd_dostupnost_val,
        'parkování': parkovani_val,
        'výtah': vytah_val,
        'město_encoded': mesto_encoded_val,
        'typ_nemovitosti_encoded': typ_nemovitosti_encoded
    }
    return features_dict


FEATURES_ORDER = [
    'rozloha_užitná', 'počet_koupelen', 'počet_pokojů', 'garáž', 'balkón', 'balkón_plocha', 
    'lodžie', 'lodžie_plocha', 'mhd_dostupnost', 'parkování', 'sklep', 'sklep_plocha', 
    'terasa', 'terasa_plocha', 'výtah', 'dispozice_numeric', 'energetická_třída_numeric', 
    'město_encoded', 'typ_nemovitosti_encoded'
]

@st.cache_resource
def load_model():

    return joblib.load('xgboost_real_estate_model.pkl')


def predict_price(model, features_dict):

    feature_array = [features_dict.get(feature, 0) for feature in FEATURES_ORDER]

    X = np.array(feature_array).reshape(1, -1)

    predicted_price = model.predict(X)[0]
    return predicted_price


def run_app():
    st.set_page_config(page_title="Predikce ceny nemovitosti", layout="centered")

    st.title("🏠 Predikce ceny nemovitosti")
    st.write("Webovou aplikaci a Model odhadu cen nemovitostí s vlastními daty vytvořil Viktor Vítovec 2025")
    st.subheader("Vyplňte parametry nemovitosti pro odhad ceny.")

    col1, col2 = st.columns(2)
    
    with col1:
        rozloha_uzitna = st.number_input('Užitná plocha (m²)', min_value=10, max_value=500, value=75, step=1)
        dispozice = st.selectbox('Dispozice', ['1+kk', '1+1', '2+kk', '2+1', '3+kk', '3+1', '4+kk', '4+1', '5+kk', '5+1'])
        energeticka_trida = st.selectbox('Energetická třída', ['A', 'B', 'C', 'D', 'E', 'F', 'G'], index=2)
        typ_nemovitosti = st.radio('Typ nemovitosti', ['Byt', 'Dům'])
    
    with col2:
        mhd_dostupnost = st.number_input('MHD dostupnost (minuty)', min_value=0, max_value=100, value=10, step=1)
        parkovani = st.number_input('Počet parkovacích míst', min_value=0, max_value=10, value=1, step=1)
        mesto_input = st.text_input('Město')
        m_code = CITY_MAP.get(mesto_input.strip(), 184)

    st.subheader("Dodatečné prvky")
    
    col3, col4 = st.columns(2)
    
    with col3:
        balkon = st.checkbox('Balkón')
        balkon_plocha = 0.0
        if balkon:
            balkon_plocha = st.number_input('Plocha balkónu (m²)', min_value=0.0, max_value=50.0, value=5.0, step=0.5)
            
        lodzie = st.checkbox('Lodžie')
        lodzie_plocha = 0.0
        if lodzie:
            lodzie_plocha = st.number_input('Plocha lodžie (m²)', min_value=0.0, max_value=50.0, value=5.0, step=0.5) 
            
        garaz = st.checkbox('Garáž')
    
    with col4:
        sklep = st.checkbox('Sklep')
        sklep_plocha = 0.0
        if sklep:
            sklep_plocha = st.number_input('Plocha sklepa (m²)', min_value=0.0, max_value=50.0, value=5.0, step=0.5)
            
        terasa = st.checkbox('Terasa')
        terasa_plocha = 0.0
        if terasa:
            terasa_plocha = st.number_input('Plocha terasy (m²)', min_value=0.0, max_value=100.0, value=10.0, step=0.5)

        vytah = st.checkbox('Výtah')

    if st.button('🔮 Predikovat cenu', type="primary"):
        try:
            model = load_model()

            features = preprocess_user_inputs(
                rozloha_uzitna=rozloha_uzitna,
                dispozice=dispozice,
                energeticka_trida=energeticka_trida,
                balkon=balkon,
                balkon_plocha=balkon_plocha,
                lodzie=lodzie,
                lodzie_plocha=lodzie_plocha,
                sklep=sklep,
                sklep_plocha=sklep_plocha,
                terasa=terasa,
                terasa_plocha=terasa_plocha,
                garaz=garaz,
                mhd_dostupnost=mhd_dostupnost,
                parkovani=parkovani,
                vytah=vytah,
                mesto_encoded=m_code,
                typ_nemovitosti_str=typ_nemovitosti,
            )

            predicted_price = predict_price(model, features)

            st.success(f"💰 **Predikovaná cena nemovitosti: {predicted_price:,.0f} Kč**")
            
            if rozloha_uzitna > 0:
                price_per_m2 = predicted_price / rozloha_uzitna
                st.info(f"📏 Cena za m²: {price_per_m2:,.0f} Kč/m²")
            
        except FileNotFoundError:
            st.error("❌ Model nebyl nalezen! Ujistěte se, že soubor 'xgboost_real_estate_model.pkl' existuje v pracovním adresáři.")
        except Exception as e:
            st.error(f"❌ Nastala chyba při predikci: {str(e)}")

if __name__ == "__main__":
    run_app()
