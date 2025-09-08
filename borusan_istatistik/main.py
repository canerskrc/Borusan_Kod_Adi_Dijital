import random
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

n = 1000
brands = ["BMW", "Audi", "Mercedes", "Toyota", "Hyundai", "RangeRover", "Mini","Ford" ]
fuel_type = ["Benzin", "Dizel", "Elektrik", "Hibrit"]
colors = ["Siyah", "Beyaz", "Gri", "Mavi", "Kırmızı"]
transmission = ["Manuel", "Otomatik"]
regions =["İstanbul","Ankara","İzmir","Bursa","Adana"]
data = []

for i in range(n):

    brand = random.choice(brands)

    model_year = random.randint[2013, 2024]

    km = random.randint[10_000, 200_000]

    fuel = random. choice(fuel_type)

    transmission = random. choice(transmission)

    color = random. choice(colors)

    region = random. choice(regions)

    engine_size = min (max (random.gauss (1.6,0.5),0.9),4.5)
    owners = random.randint(1,4)
    accidents = np.random.poisson(0.5)


    #Fiyat Hesaplama Modeli

    brand_factor ={
        "BMW":1.3, "Audi":1.25, "Mercedes":1.4,"Toyota":1.0, "Hyundai":0.8,
        "RangeRover":1.6, "Mini":1.4, "Ford":1.2
    }

    base_price=(
        500_000
        - (2024 - model_year) *1500
        - (km / 1000) *100
        + brand_factor[brand] * 20_000
        + engine_size * 5000
        - accidents * 1000
        + random.gauss(0,5000)
    )


    # 2016 model araç için
    # 8 yıl * 1500 TL fiyat düşecek

    price = int(min(max(base_price, 1_000_000),6_000_000))

    data.append([brand,model_year,km,fuel,transmission,color,region,owners,accidents,price,round(engine_size,1)])

sutun_isimleri = ["Brands","ModelYear", "KM", "Fuel","Transmission",
          "Color","Region","Owners","Accidents","Price","EngineSize"]

df_cars = pd.DataFrame(data, columns= sutun_isimleri)
df_cars.head()
# KORELASYON ANALİZİ

km_val = df_cars["KM"]
price_val = df_cars["Price"]



# PEARSON İLİŞKİ = +1 ile -1 aralığında bir değer üretilir.
#pearson_corr = df_num.corr(method="pearson")
pearson_corr, p_val_pearson = pearsonr(km_val, price_val)

pearson_matrix = df_cars.corr(method="pearson")
#ikinci yöntem
pearson_corr = df_cars["KM"].corr(df_cars["Price"], method="pearson")
#formül yazımı
x = df_cars["KM"]
y = df_cars["Price"]

x_mean = x.mean()
y_mean = y.mean()

cov_xy = ((x-x_mean)*(y-y_mean)).sum() / (len(x)-1)
std_x = np.sqrt(((x-x_mean) **2).sum() / (len(x)-1))
std_y = np.sqrt(((y-y_mean) **2).sum() / (len(y)-1))

pearson_manuel = cov_xy / (std_x * std_y)

print("Pearson formülü ile : ", pearson_manuel)

X^2 = 25.4
p= 0.0001
df = 3

#Spearman Korelasyonu ( sıralama ilişkisi ) iki değişken arasındaki
# ilişkinin var olup olmadığına bakar. Lineer olup olmamasıyla ilgilenmez.

spearman_corr, p_val_spearman = spearmanr(km_val, price_val)

print("Spearman correlation ile : ", spearman_corr, "p-değeri": p_val_spearman)

#Görselleştirme

plt.figure(figsize = (8,5))
plt.scatter(km_val,price_val, alpha = 0.4)
plt.xlabel("KM")
plt.ylabel("Price")
plt.title("Fiyat-KM İlişkisi")
plt.show()




#Z Score ( Outlier Tespiti)

#z = ((x - X') / s)

z_scores = (df_cars["Price"] - df_cars["Price"].mean()) / df_cars["Price"].std()
outliers = df_cars[np.abs(z_scores) > 3]

# Regresyon Çizgisi ( Lineer Trend)

#import seaborn as sns
sns.regplot(x="KM", y="Price", data=df_cars, line_kws={"color":"red"})



