# RNA-seq AI Builder — Lung Cancer Classifier

Pipeline สำหรับสร้าง ML model จำแนกมะเร็งปอด โดยใช้ข้อมูล RNA-seq ระดับ exon
รันบน **WSL (Ubuntu)** เพราะ tools ส่วนใหญ่ทำงานบน Linux

---

## ภาพรวม Pipeline

```
FASTQ Files
    │
    ▼
Step 1: QC ก่อน Trim          (qc1/)
    │
    ▼
Step 2: Trim Adapter/Low-quality  (trim/)
    │
    ▼
Step 3: QC หลัง Trim          (qc2/)
    │
    ▼
Step 4: Download Genome + Annotation  (Download genome + annotation/)
    │
    ▼
Step 5: Build Genome Index     (STAR/)
    │
    ▼
Step 6: Align Reads → Genome  (STAR/)
    │
    ▼
Step 7: Count Exons per Sample (exon x samples/)
    │
    ▼
Step 8: Merge เป็น Matrix     (matrix/)
    │
    ▼
Step 9: Preprocess + Feature Selection  (preprocess/)
    │
    ▼
Step 10: Train ML Model        (train/)
    │
    ▼
Result: Confusion Matrix + ROC Curve  (result/)
```

---

## โครงสร้างโฟลเดอร์

```
rna_ai_builder/
├── 00_download_fastq/        # วิธี download ข้อมูล FASTQ
├── 01_download_reference/    # Script download genome + GTF annotation
├── 02_qc_before_trim/        # ผล FastQC ก่อน trim
├── 03_trim/                  # Script และผล trimming (fastp)
├── 04_qc_after_trim/         # ผล FastQC หลัง trim
├── 05_align/                 # Script build genome index และ align (STAR)
├── 06_count_exons/           # Script count exons (featureCounts)
├── 07_merge_matrix/          # Script merge count files เป็น matrix
├── 08_preprocess/            # Script feature selection
├── 09_train/                 # Script train model
├── 10_result/                # รูปผลลัพธ์ (confusion matrix, ROC)
└── pipeline/                 # Pipeline scripts (trim → align → count)
    ├── run_ERR_dataset.sh    #   ERR164xxx — 10 cancer + 12 non_cancer
    └── run_SRR_dataset.sh    #   SRR24166xxx — 50 tumor + 50 normal
```

---

## รายละเอียดแต่ละขั้นตอน

### Step 1 — QC ก่อน Trim
**โฟลเดอร์:** `qc1/`  
ใช้ **FastQC** ตรวจสอบคุณภาพ raw reads เพื่อดูว่ามี adapter contamination หรือ quality ต่ำตรงไหน

```bash
bash 02_qc_before_trim/run_fast.sh
```

---

### Step 2 — Trim Adapter + Low-quality Bases
**โฟลเดอร์:** `trim/`  
ใช้ **fastp** ตัด adapter และ bases คุณภาพต่ำออก

```bash
bash 03_trim/trim.sh
```

> ตัวอย่าง: ERR164550 — reads จาก 38.7M → 26.6M หลัง trim

---

### Step 3 — QC หลัง Trim
**โฟลเดอร์:** `qc2/`  
รัน FastQC อีกครั้งหลัง trim เพื่อยืนยันว่าข้อมูลสะอาดแล้ว

```bash
bash 04_qc_after_trim/run_fast.sh
```

---

### Step 4 — Download Genome + Annotation
**โฟลเดอร์:** `Download genome + annotation/`

| ไฟล์ | คืออะไร |
|------|---------|
| Genome (FASTA) | แผนที่ DNA มนุษย์ทั้งหมด (A/T/G/C ของทุก chromosome) |
| Annotation (GTF) | ป้ายชื่อ บอกว่าตำแหน่งไหนในแผนที่คือยีนอะไร exon ไหน |

ใช้ reference: **Homo sapiens GRCh38 / Ensembl 113**

```bash
bash 01_download_reference/source.sh
bash 01_download_reference/unzip.sh
```

---

### Step 5 — Build Genome Index
**โฟลเดอร์:** `STAR/`  
จัดระเบียบ genome ใหม่เพื่อให้ STAR ค้นหาได้เร็ว (ทำครั้งเดียว)

```bash
bash 05_align/build_genome_index.sh
```

---

### Step 6 — Align Reads → Genome
**โฟลเดอร์:** `STAR/`  
ใช้ **STAR** (Spliced Transcripts Alignment to a Reference) จับคู่ reads กลับไปยังตำแหน่งในจีโนม ผลลัพธ์คือไฟล์ BAM

```bash
bash 05_align/Align.sh
```

---

### Step 7 — Count Exons
**โฟลเดอร์:** `exon x samples/`  
ใช้ **featureCounts** นับจำนวน reads ที่ตกในแต่ละ exon ของแต่ละ sample

```bash
bash "06_count_exons/exon-level count.sh"
```

---

### Step 8 — Merge เป็น Matrix
**โฟลเดอร์:** `matrix/`  
รวม count files ของทุก sample เข้าเป็นตาราง (exons × samples)

```bash
# SRR samples เดี่ยว
python3 07_merge_matrix/merge_to_matrix.py

# SRR + ERR รวมกัน
python3 07_merge_matrix/merge_combined.py
```

---

### Step 9 — Preprocess + Feature Selection
**โฟลเดอร์:** `preprocess/`  
ตอนนี้มี ~2,164,410 features (exons) ซึ่งเยอะเกินไป จึงทำ feature selection 3 ขั้น:

| ขั้นตอน | วิธี | ผล |
|---------|------|----|
| 1. Filter | เอา exon ที่มี count > 0 ใน ≥10% ของ samples ออกมา | ลด noise |
| 2. Normalize | log1p transform | สเกลข้อมูลไม่ต่างกันมากเกินไป |
| 3. Select | top 5,000 exons by variance | เหลือ features ที่มีประโยชน์ |

```bash
# SRR เดี่ยว
python3 08_preprocess/preprocess.py

# SRR + ERR
python3 08_preprocess/preprocess_combined.py
```

---

### Step 10 — Train ML Model
**โฟลเดอร์:** `train/`  
Train 3 models พร้อมกัน: **Random Forest**, **XGBoost**, **SVM**  
เลือก 3 models ที่มีวิธีคิดต่างกัน — ถ้าทั้ง 3 ตัวให้ผลใกล้กัน แสดงว่าไม่ใช่ความบังเอิญ

**ขั้นตอน:**
1. Train เฉพาะ SRR samples ก่อน (ข้อมูลเยอะกว่า)
2. Cluster ERR + SRR เพื่อหา batch effect → พบว่าผลค่อนข้างแย่
3. แก้ด้วย **ComBat** (batch correction)
4. Train model รวม ERR + SRR + ComBat แล้วเปรียบเทียบกับ SRR เดียว

```bash
# SRR เดี่ยว
python3 "09_train/train_model(only_srr).py"

# SRR + ERR + ComBat
python3 09_train/train_model_combined.py
```

> ผลต่างกันแค่ ±2% → batch correction ได้ผล

---

## ผลลัพธ์
**โฟลเดอร์:** `result/`

| ไฟล์ | คืออะไร |
|------|---------|
| `10_result/confusion_matrices.png` | Confusion matrix (SRR only) |
| `10_result/confusion_matrices_combined.png` | Confusion matrix (SRR + ERR + ComBat) |
| `10_result/roc_curves.png` | ROC curve (SRR only) |
| `10_result/roc_curves_combined.png` | ROC curve (SRR + ERR + ComBat) |

---

## Run Pipeline ทั้งหมดในคราวเดียว

`run_pipeline.sh` รวม trim + align + count ไว้ในสคริปต์เดียว รองรับทุก sample

```bash
# ERR dataset (10 cancer + 12 non_cancer)
bash pipeline/run_ERR_dataset.sh

# SRR dataset (50 tumor + 50 normal)
bash pipeline/run_SRR_dataset.sh
```

---

## Tools ที่ใช้

| Tool | ทำอะไร |
|------|--------|
| FastQC | Quality check |
| fastp | Trimming |
| STAR | RNA-seq alignment |
| featureCounts (subread) | Exon counting |
| Python (pandas, scikit-learn) | Matrix merge, preprocess, train |
| XGBoost | ML model |
| pyComBat | Batch effect correction |

---

## ข้อมูล

- รันบน WSL (Ubuntu) บน Windows
- Reference genome: Homo sapiens GRCh38, Ensembl annotation v113
- Samples: ERR164xxx (cancer/non-cancer), SRR series
