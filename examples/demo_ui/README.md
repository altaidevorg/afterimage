# Afterimage — Demo arayüzü kullanım kılavuzu

Bu arayüz, tarayıcıda çalışan bir **demo uygulamasıdır**. Amaç: metin veya dosya olarak verdiğiniz bağlama dayalı **örnek konuşmalar ve veri setleri** üretmek; gerekirse **araç (tool) çağrıları** ve **model eğitimi** adımlarını denemek.

Uygulamayı açtığınızda üst menüden sayfalar arasında geçebilirsiniz. Aşağıda her menü başlığında neler yapabileceğiniz özetlenir.

**Not:** Uygulamayı size bir bağlantı veya adres ile veren ekip / yönetici, gerekli hesap ve erişimleri de sağlar. Bir şey çalışmazsa onlara başvurun.

---

## How it Works (Nasıl çalışır?)

Burada sistemin genel mantığı anlatılır: belgelerin yüklenmesi, farklı kullanıcı tipleri (persona), soru üretimi ve yanıtlar. Küçük bir akış şeması da vardır.

Bu sayfada veri üretmezsiniz; sadece **bilgi** içindir.

---

## Generic Conversation (Genel konuşma)

**Ne işe yarar:** Verdiğiniz metinlere dayalı, doğal konuşma tarzında **sentetik diyalog örnekleri** üretir.

**Nasıl kullanılır:**

1. **Bağlam:** Hazır metni düzenleyebilir veya kendi metninizi yazabilir; isteğe bağlı olarak dosya yükleyebilirsiniz (metin, tablo veya belge türleri menüde listelenir).
2. **Ayarlar:** Asistanın rolünü tanımlayan metin ve üretilecek **diyalog sayısı**.
3. **Üret:** Butona bastığınızda tabloda soru–cevap ve ilgili bilgiler birikir; işlem bitince veriyi **indirebilirsiniz**.

---

## Structured Generation (Yapılandırılmış üretim)

**Ne işe yarar:** Müşteri hizmeti benzeri senaryolarda **aynı tür alanları** (örneğin niyet, aciliyet, düşünce özeti, yanıt) dolduran tutarlı örnekler üretir.

**Nasıl kullanılır:** Bağlamı manuel veya dosyadan verin, asistan rolünü ve **örnek sayısını** seçin, üret butonuna basın. Sonuçları tabloda görür, bitince **dosya olarak indirebilirsiniz**.

---

## Tool Calling (Araç çağrıları)

**Ne işe yarar:** Doğal dil isteklerinin, tanımlı **akıllı araçlara** (örneğin ışık açma, termostat) nasıl eşleneceğine dair örnek veri üretir. İsterseniz üretim bittikten sonra **model eğitimini** de başlatabilirsiniz.

**Adımlar (sihirbaz):**

1. **Bağlam:** Örnek akıllı ev metni veya kendi belgeniz.
2. **Ayarlar:** Asistan rolü, örnek sayısı ve veri setinin kütüphanede hangi **kategori** altında görüneceği.
3. **Araç seçimi:** Araç Kütüphanesi’ndeki araçları kategorilere göre işaretleyin; tümünü seç / kaldır ve yenile seçenekleri vardır.
4. **Üret (ve isteğe bağlı eğit):** Örnekleri oluşturur; **“Train Model after generation”** işaretliyse ardından eğitim ekranı açılır ve ilerleme gösterilir. Uygun olduğunda eğitilmiş modeli **indirebilirsiniz**.

---

## Train Model (Model eğitimi)

**Ne işe yarar:** Daha önce üretilmiş veya yüklenmiş **veri setlerinizi** seçip birleştirmenizi, filtrelemenizi ve bir yapay zeka modelini **eğitmenizi** sağlar; ardından değerlendirme ve basit **sohbet** adımları vardır.

**1. Veri kütüphanesi**

- Kategorilere göre listelenen setleri seçebilir, yenileyebilirsiniz.
- Özet ekranda istatistik ve araç kullanımına göre dağılım görünür.
- **Araç filtresi:** Hangi araçtan kaç örneğin eğitime gireceğini kaydırıcılarla sınırlayabilirsiniz.
- Setleri **yeniden adlandırma**, **kategori değiştirme**, **birleştirme**, **araç gruplarına göre bölme** veya **silme** (onaylı) yapılabilir.

**2. Eğitim**

- **Normal mod:** Hazır ayarlarla eğitim; durum ve ilerleme gösterilir.
- **Geliştirici modu:** Daha teknik kullanıcılar için ek parametreler ve ayrıntılı günlük metni.

Eğitim tamamlanınca modeli **paket olarak indirme** seçeneği sunulabilir.

**3. Değerlendirme** — Modelin bir test üzerinden ölçülmesi; sonuçlar ekranda metin olarak görünür.

**4. Sohbet** — Eğitilen modelle kısa mesajlaşma denemesi.

---

## Tool Library (Araç kütüphanesi)

**Ne işe yarar:** Tool Calling sayfasında kullanılacak **araç tanımlarını** görüntüleme, düzenleme ve yenilerini ekleme.

- Solda kategorilere göre liste; **Yenile** ve **Yeni araç** ile başlarsınız.
- Sağda bir aracı seçtiğinizde ayrıntılar, kategori değişikliği, düzenleme ve silme.
- Yeni araç eklerken:
  - **Dış bağlantıdan içe aktarma** (kurulu bir araç sunucusu varsa),
  - **Elle** isim, açıklama ve parametrelerle tanımlama,
  - **Koddan** örnek bir fonksiyon yapıştırarak tanımlama  
  seçenekleri bulunur.

Hazır (yerleşik) araçlar listede ayrıca belirtilir.

---

## Birlikte kullanılan bölümler

Çoğu üretim sayfasında **Bağlam kaynağı** (elle yazma veya dosya yükleme) ve **asistan rol metni** vardır. Üretim sırasında tablo canlı güncellenir; bittiğinde çoğu akışta veri setini **indirebilirsiniz**.

Eğitim ve model indirme işlemleri **zaman alabilir** ve sunucunun internete bağlı olmasını gerektirir.
