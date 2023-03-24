#############################################
# PROJE: Association Rule Based Recommender System
#############################################
#İş Problemi
#Aşağıda 3 farklı kullanıcının sepet bilgileri verilmiştir. Bu sepet bilgilerine en
#uygun ürün önerisini birliktelik kuralı kullanarak yapınız. Ürün önerileri 1 tane
#ya da 1'den fazla olabilir. Karar kurallarını 2010-2011 Germany müşterileri üzerinden türetiniz.
#Kullanıcı 1’in sepetinde bulunan ürünün id'si: 21987
#Kullanıcı 2’in sepetinde bulunan ürünün id'si : 23235
#Kullanıcı 3’in sepetinde bulunan ürünün id'si : 22747
#
#
#Veri Seti Hikayesi
#Online Retail II isimli veri seti İngiltere merkezli bir perakende şirketinin 01/12/2009 - 09/12/2011 tarihleri arasındaki online satış
#işlemlerini içeriyor. Şirketin ürün kataloğunda hediyelik eşyalar yer almaktadır ve çoğu müşterisinin toptancı olduğu bilgisi
#mevcuttur.

#InvoiceNo Fatura Numarası ( Eğer bu kod C ile başlıyorsa işlemin iptal edildiğini ifade eder )
#StockCode Ürün kodu ( Her bir ürün için eşsiz )
#Description Ürün ismi
#Quantity Ürün adedi ( Faturalardaki ürünlerden kaçar tane satıldığı)
#InvoiceDate Fatura tarihi
#UnitPrice Fatura fiyatı ( Sterlin )
#CustomerID Eşsiz müşteri numarası
#Country Ülke ismi



#Görev 1: Veriyi Hazırlama

import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False) # çıktının tek bir satırda olmasını sağlar.
from mlxtend.frequent_patterns import apriori, association_rules

#Adım 1: Online Retail II veri setinden 2010-2011 sheet’ini okutunuz.
df_ = pd.read_excel("Datasets/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()
df.head()
df.describe().T
# quantitiy ve price da (-) değerler var. bunun nedeni faturadaki iadeler.
# Qua. min-max, çeyreklik değerler arası baya açık. aykırı değerler var.
#
df.isnull().sum() #Description ve CustomerID de eksik gözlemler var.
df.shape
#Adım 2: StockCode’u POST olan gözlem birimlerini drop ediniz. (POST her faturaya eklenen bedel, ürünü ifade etmemektedir.)
df[df["StockCode"] == "POST"]
#Adım 3: Boş değer içeren gözlem birimlerini drop ediniz.
#Adım 4: Invoice içerisinde C bulunan değerleri veri setinden çıkarınız. (C faturanın iptalini ifade etmektedir.)
#Adım 5: Price değeri sıfırdan küçük olan gözlem birimlerini filtreleyiniz.
#Adım 6: Price ve Quantity değişkenlerinin aykırı değerlerini inceleyiniz, gerekirse baskılayınız.

df.isnull().sum()
df.describe().T

# aykırı değerleri bastıralım: IQR hesabı
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01) # %1 çeyrek değerini al bunu quartile1 olarak tut.
    quartile3 = dataframe[variable].quantile(0.99) # %99 çeyrek değerini al bunu quartile3 olarak tut.
    #normalde quartile çeyrek hesabında 0.25-0.75 çeyreklikler kullanılır. ucundan dokunmak istediği için. VK nın yorumu böyle.
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range # 99luk çeyrekten 1,5 IQR uzaklıktaki nokta benim üst değerimdir.(up limit)
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit # df teki ilgili variable ı seç, belirlenen low limitten aşağıda olanları getir. low_limitle değiştir.
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#hepsini bir arada çözümleyen fonksiyon.
def retail_data_prep(dataframe):
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["StockCode"].str.contains("POST", na=False)]
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    return dataframe

df = retail_data_prep(df)
df.isnull().sum()
df.describe().T

# Görev 2: Alman Müşteriler Üzerinden Birliktelik Kuralları Üretme
# Adım 1: Aşağıdaki gibi fatura ürün pivot table’i oluşturacak create_invoice_product_df fonksiyonunu tanımlayınız.

#Description NINE DRAWER OFFICE TIDY SET 2 TEA TOWELS I LOVE LONDON SPACEBOY BABY GIFT SET…
#Invoice
#536370         0                       1                                0
#536852         1                       0                                1
#536974         0                       0                                0
#537065         1                       0                                0
#537463         0                       0                                1

df_g = df[df['Country'] == "Germany"]

df_g.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).head(20)
df_g.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().iloc[0:5, 0:5]
df_g.groupby(['Invoice', 'Description']).agg({"Quantity": "sum"}).unstack().fillna(0).iloc[0:5, 0:5]
df_g.groupby(['Invoice', 'Description']). \
    agg({"Quantity": "sum"}). \
    unstack(). \
    fillna(0). \
    applymap(lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]

def create_invoice_product_df(dataframe, id=False):
    if id: # Eğer id= True ise yukarıda "description" a göre yapılan işlemi bu sefer "StockCode" a göre yap, return et. # False ise else'e git. #Description a göre çalıştır.
        return dataframe.groupby(['Invoice', "StockCode"])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)
    else:
        return dataframe.groupby(['Invoice', 'Description'])['Quantity'].sum().unstack().fillna(0). \
            applymap(lambda x: 1 if x > 0 else 0)

g_inv_pro_df = create_invoice_product_df(df_g)

g_inv_pro_df = create_invoice_product_df(df_g, id=True) #stockcodelara göre birliktelik kurallarını çıakrıcaz

# stock kodu ile ürün tanımını nasıl eşleştirebilirim: yukarıda fonk. stock koduna göre çalıştırdım ama ürünün ne olduğunu bilmiorum dediğim noktada:
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist() #values[0] sadece strign değere ulaşmak istiyoruz. .tolist() listeye dönüştür diyoryz.
    print(product_name)

df["StockCode"]
check_id(df_g, 22899)
#22899 nolu stock kodunun ürün tanımı neymiş? ["CHILDREN'S APRON DOLLY GIRL "]


# Adım 2: Kuralları oluşturacak create_rules fonksiyonunu tanımlayınız ve alman müşteriler için kurallarını bulunuz.

frequent_itemsets = apriori(g_inv_pro_df, #apriori derki, yukarıda verileri işlediğimiz fonk. ismi üzerinden,dfteki aynı isimleri kullanmak istersen use_colnames=True yap. bende sana olası ürün birlikteliklerinin çıktılarını vereyim.
                            min_support=0.01, #teori kısmında min support değeri veriyorduk. bu o. support değeri %1in altında olmasın.
                            use_colnames=True)

frequent_itemsets.sort_values("support", ascending=False) #supporta göre azalan şekilde sırala.

#       support                      itemsets
#389   0.249443                      22326 stok kodlu ürünün  tek başına görülme olasılığı 0.2494
#387   0.160356                      (22328)

#bizim tam ihtiyacımız olan bu değil
#birliktelik kurallarını çıkarmamız gerekiyor.
rules = association_rules(frequent_itemsets,
                          metric="support",
                          min_threshold=0.01)

#rules
#Out[206]:
#      antecedents                          consequents  antecedent support  consequent support   support  confidence       lift  leverage  conviction
#0         (16237)                              (22326)            0.011136            0.249443  0.011136    1.000000   4.008929  0.008358         inf
#1         (22326)                              (16237)            0.249443            0.011136  0.011136    0.044643   4.008929  0.008358    1.035073
#2         (20674)                              (20675)            0.022272            0.033408  0.013363    0.600000  17.960000  0.012619    2.416481

# antecedents: önceki ürün (16237)
# consequents: ikinci ürün (22326)
# antecedent support: ürünün tek başına gözlenme olasılığı (16237) ürününün 0.011136
# consequent support: 2.ürünün tek başına gözlenme olasılığı (22326) ürününün 0.249443
# support:  antecedents: önceki ürün ve consequents: ikinci ürünün birlikte gözükme olasılığı 0.011136
# confidence: X ürünü alndığında Y nin alınması olasılığı  1.000000
# lift:  X ürünü satın alndığında Y nin satın alınması olasılığı 4.008929 kat artar
# leverage: kaldıraç etkisi. lifte benzer. leverage değeri supportu yüksek olan değerlere öncelik verme egiliminde bundan dolayı ufak bir yanlılığı vardır.
# Lift değeri ise daha az sıklıkta olmasına ragmen bazı ilişkileri yakalayabilmektedir. bizim için daha değerlidir. yansızdır.
# conviction: Y ürünü olmadan X ürünün beklenen değeri, frekansıdır. diğer taraftan X ürünü olmadan Y ürünün beklenen frekansıdır.
#leverage ve conf.  değerine çok odaklanmıyoruz. lift conf. ile gidiyoruz.

#Adım 1: check_id fonksiyonunu kullanarak verilen ürünlerin isimlerini bulunuz.

def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist() #values[0] sadece strign değere ulaşmak istiyoruz. .tolist() listeye dönüştür diyoryz.
    print(product_name)

df["StockCode"]
check_id(df_g, 22326)

# ilk index için:
# antecedents: önceki ürün (16237) ['SLEEPING CAT ERASERS']
# consequents: ikinci ürün (22326) ['ROUND SNACK BOXES SET OF4 WOODLAND ']
# antecedent support: ürünün tek başına gözlenme olasılığı (16237) ['SLEEPING CAT ERASERS'] ürününün 0.011136
# consequent support: 2.ürünün tek başına gözlenme olasılığı (22326) ['ROUND SNACK BOXES SET OF4 WOODLAND '] ürününün 0.249443
# support:  antecedents: önceki ürün ve consequents: ikinci ürünün birlikte gözükme olasılığı 0.011136
# confidence: (16237) ['SLEEPING CAT ERASERS'] ürünü alndığında ['ROUND SNACK BOXES SET OF4 WOODLAND '] nin alınması olasılığı  1.000000
# lift:  ['SLEEPING CAT ERASERS'] ürünü satın alndığında ['ROUND SNACK BOXES SET OF4 WOODLAND '] nin satın alınması olasılığı 4.008929 kat artar


# kullanıcı sepetine (16237) stok kodlu ['SLEEPING CAT ERASERS'] ürününü eklediğinde (22326) ['ROUND SNACK BOXES SET OF4 WOODLAND '] bu ürünü önericez.


#Adım 2: arl_recommender fonksiyonunu kullanarak 3 kullanıcı için ürün önerisinde bulununuz.

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]

#arl_recommender(rules_df, product_id, rec_count=1)

arl_recommender(rules, 20674, 3)

#['GREEN POLKADOT BOWL'] ürününe 3 adet birlikrte satın alınabilecek ürün

df.head()
#Adım 3: Önerilecek ürünlerin isimlerine bakınız.
#[21240, 21245, 20675]
check_id(df_g, 20674)
#['BLUE POLKADOT CUP']
#['GREEN POLKADOT PLATE ']
#['BLUE POLKADOT BOWL']

# İş problemindeki asıl istenen 3 kullanıcıya önerilecek isimler:

#Kullanıcı 1’in sepetinde bulunan ürünün id'si: 21987
check_id(df_g, 21987) # ['PACK OF 6 SKULL PAPER CUPS'] sepetinde bu ürün var.
arl_recommender(rules, 21987, 1) #21989 stok kodlu ürünü önerdi.
check_id(df_g, 21989) #['PACK OF 20 SKULL PAPER NAPKINS']

#Kullanıcı 2’in sepetinde bulunan ürünün id'si : 23235
check_id(df_g, 23235) # ['STORAGE TIN VINTAGE LEAF'] sepetinde bu ürün var.
arl_recommender(rules, 23235, 1) #[23244] stok kodlu ürünü önerdi.
check_id(df_g, 23244) #['ROUND STORAGE TIN VINTAGE LEAF']


#Kullanıcı 3’in sepetinde bulunan ürünün id'si : 22747
check_id(df_g, 22747)  #["POPPY'S PLAYHOUSE BATHROOM"] sepetinde bu ürün var.
arl_recommender(rules, 22747, 1) #[22746] stok kodlu ürünü önerdi.
check_id(df_g, 22746) # ["POPPY'S PLAYHOUSE LIVINGROOM "]
