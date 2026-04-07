# Afterimage Demo UI

Gradio tabanlı arayüz: sentetik veri üretimi (konuşma, yapılandırılmış çıktı, tool calling), araç kütüphanesi yönetimi ve isteğe bağlı model eğitimi / değerlendirme / sohbet.

## Gereksinimler

- Python 3.11+
- Proje kökünden kurulum: `pip install -e .` (veya `uv sync`)
- **Eğitim** (Train Model, Tool Calling’de “Train after generation”): `pip install -e ".[training]"` — `trl`, `torch`, `transformers` vb.
- Ortam değişkenleri (köke `.env` önerilir):
  - `DEEPSEEK_API_KEY` — demo üretim akışları (zorunlu)
  - `GEMINI_API_KEY` — kütüphane örnekleri için isteğe bağlı
  - `HF_TOKEN` — Hugging Face model indirme ve eğitim için

Çalıştırmadan önce anahtarları shell’e aktarın:

```bash
cd /path/to/afterimage
set -a && source .env && set +a
python examples/demo_ui/app.py
```

Varsayılan adres: `http://127.0.0.1:7860` (uygulama `share=True` ile public Gradio linki de üretebilir).

---

## Sayfalar ve rotalar

Üst gezinme çubuğundan sayfalar arasında geçilir. Aşağıda her rota ve o sayfada yapılabilecekler özetlenir.

### How it Works (`/` — ana sayfa)

- Afterimage boru hattının kısa açıklaması: bağlam yükleme → persona → talimat → yanıt / şema.
- Mermaid diyagramı (iş akışı özeti).
- Bu demodaki üç üretim türünün (konuşma, tool calling, yapılandırılmış üretim) ne işe yaradığına dair metin.

Burada üretim başlatılmaz; bilgilendirme amaçlıdır.

---

### Generic Conversation (`/conversations`)

**Amaç:** Belgelere dayalı, çok dönüşlü konuşma tarzı sentetik veri üretmek.

**Ne yapılır:**

1. **Context Source** — Manuel metin veya dosya yükleme (`.txt`, `.csv`, `.tsv`, `.jsonl`, `.docx`, `.rtf`, `.html`). Yapılandırılmış dosyalarda içerik sütunu seçilir (`key`).
2. **Configuration** — Respondent sistem prompt’u ve üretilecek **diyalog sayısı** (1–50).
3. **Generate Conversations** — Persona + talimat + yanıt üretimi; tabloda Instruction, Response, Context, Persona vb. kolonlar.
4. Tamamlanınca **JSONL indirme** üretilen veri seti için kullanılabilir.

---

### Structured Generation (`/structured`)

**Amaç:** Sabit bir şema (ör. müşteri desteği senaryosu) için tutarlı, alanları doldurulmuş örnekler üretmek.

**Ne yapılır:**

1. **Context Source** — Manuel veya dosya (varsayılan örnek bağlamlar TechGadget politikası / troubleshooting tarzıdır).
2. **Configuration** — Respondent prompt ve **örnek sayısı** (1–50).
3. **Generate Structured Data** — `CustomerSupportInteraction` benzeri alanlar: Persona, Instruction, Intent, Urgency, Reasoning, Response vb.
4. **İndirme** ile JSONL alınır.

---

### Tool Calling (`/tools`)

**Amaç:** Seçilen araç şemalarına uygun, doğal dil → tool çağrısı veri seti üretmek (ör. akıllı ev asistanı). İsteğe bağlı üretim sonrası model eğitimi.

**Sihirbaz adımları:**

| Adım | İçerik |
|------|--------|
| **1. Context Source** | Manuel veya dosya; varsayılan akıllı ev kullanım kılavuzu metni. |
| **2. Configuration** | Respondent prompt, örnek sayısı (1–50), **Dataset Category** (kütüphanede gruplama için). |
| **3. Select Tools** | Tool Library’deki araçlar kategorilere göre; kategori / tümünü seç, tümünü kaldır, yenile. |
| **4. Generate & Train** | İsteğe bağlı **“Train Model after generation”** — üretim bitince aynı veri ile eğitim sürecini başlatır. |

**Çıktı:** Persona, Instruction, Response, Reasoning, Tool Calls kolonları; gizli indirme bileşeni eğitim için dosya yolunu kullanır. Eğitim açıksa ilerleme ve (başarılıysa) eğitilmiş model indirme.

---

### Train Model (`/train`)

**Amaç:** `datasets/` altındaki JSONL veri setlerini seçmek, birleştirmek/filtrelemek, **Function Gemma** tabanlı ince ayar eğitimi çalıştırmak, değerlendirme ve sohbet.

**Adım 1 — Dataset Library**

- Kategorilere göre veri setleri listesi; seçim, yenileme.
- **Training Overview:** özet istatistikler, araç bazlı örnek dağılımı.
- **Filter by Tool:** Her araç için kaç örneğin eğitime dahil edileceğini sınırlayan kaydırıcılar (metadata’daki tool kullanımına göre).
- **Edit:** Veri seti adı ve kategori.
- **Merge:** Birden fazla seti tek dosyada birleştirme.
- **Split:** Araçlara göre gruplara bölerek yeni dosyalar oluşturma.
- **Delete:** Veri seti silme (onay ile).

**Adım 2 — Train Model**

- **Normal Mode:** Sabit hiperparametrelerle eğitim; durum ve ilerleme çubuğu.
- **Developer Mode:** Epoch, batch size vb. parametrelerle eğitim; ham log çıktısı.
- Başarılı eğitimde **Download** ile paketlenmiş model indirilebilir (`final_model_stable`).

**Adım 3 — Evaluate Model**

- Eğitilen modele karşı değerlendirme betiği çalıştırma; sonuçlar log alanında.

**Adım 4 — Chat with Model**

- Eğitilmiş model ile basit sohbet (chat template ile).

---

### Tool Library (`/tool-library`)

**Amaç:** Tool Calling üretiminde kullanılacak araç tanımlarını yönetmek (kategori, önizleme, düzenleme, silme).

**Sol panel:** Kategorilere göre gruplanmış araç listesi; **Refresh**, **+ New Tool**.

**Sağ panel — yeni/düzenleme:**

1. **Import from MCP** — MCP sunucusuna bağlanıp araç listesi çekme:
   - Local (komut + argümanlar), Remote (SSE URL), Config (JSON).
   - Bulunan araçlardan seçip içe aktarma.
2. **Manual Entry** — İsim, açıklama, parametre ekleme/çıkarma, **Save Tool**.
3. **From Code** — Tip ipuçlı Python fonksiyonu yapıştırıp **Parse & Preview** veya doğrudan **Save Tool**.

**Önizleme:** Araç detayı, kategori değiştirme, düzenleme, silme. Yerleşik (built-in) şemalar etiketle gösterilir.

---

## Ortak UI öğeleri

- **Context Source:** Çoğu üretim sayfasında “Manual Entry” / “File Upload”; dosya formatları `base.py` içindeki `create_context_section` ile uyumludur.
- **Respondent System Prompt:** Üretilen “asistan” davranışını tanımlar.
- Üretim sırasında **canlı tablo** ve durum mesajları; bitişte **JSONL indirme** (sayfaya göre).

---

## Sorun giderme

| Sorun | Olası neden |
|-------|-------------|
| `DEEPSEEK_API_KEY` hatası | `.env` yüklenmedi veya anahtar eksik. |
| Eğitimde `ModuleNotFoundError: trl` | `pip install -e ".[training]"` yapılmadı. |
| Eğitimde HF / model hatası | `HF_TOKEN` eksik veya geçersiz; ağ erişimi. |
| `train.py` farklı veri okudu | Runner artık hazırlanan dosyayı `--dataset` ile iletir; yine de `examples/demo_ui/training_scripts/data` altını kontrol edin. |

---

## İlgili dosyalar

- `app.py` — Rotalar ve `launch` ayarları.
- `core/` — Üretim, depolama, eğitim koşucusu.
- `training_scripts/train.py` — İnce ayar eğitimi.
- `pyproject.toml` — `[project.optional-dependencies] training` paket grubu.

Daha genel kütüphane tasarımı için depo kökündeki `DESIGN.md` dosyasına bakın.
