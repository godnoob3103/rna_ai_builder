# RNA-seq AI Builder — Cancer Classifier

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
├── 00_download_fastq/        # วิธี download ข้อมูล FASTQ และ script ดาวน์โหลด (script.ps1)
├── 01_download_reference/    # Script download genome + GTF annotation
├── 02_qc_before_trim/        # ผล FastQC ก่อน trim
├── 03_trim/                  # Script และผล trimming (fastp)
├── 04_qc_after_trim/         # ผล FastQC หลัง trim
├── 05_align/                 # Script build genome index และ align (STAR)
├── 06_count_exons/           # Script count exons (exon_level_count.sh)
├── 07_merge_matrix/          # Script merge count files เป็น matrix (merge_to_matrix.py และ merge_combined.py)
├── 08_eda/                   # การวิเคราะห์ข้อมูลเชิงสำรวจ (Exploratory Data Analysis) และผลภาพวาดความสัมพันธ์
├── 09_preprocess/            # Script feature selection (preprocess.py และ preprocess_combined.py)
├── 10_train/                 # Script เทรนโมเดล (มีทั้งโมเดลแบบ ComBat, Data Leakage correction, Y-Scrambling, error analysis และ deep learning)
├── 11_result/                # สรุปและรายงานผลลัพธ์ทั้งหมด พร้อมรายงานและข้อมูลการวิเคราะห์เชิงลึก
└── pipeline/                 # Pipeline scripts (trim → align → count)
    ├── run_ERR_dataset.sh    #   ERR164xxx — 10 cancer + 12 non_cancer
    └── run_SRR_dataset.sh    #   SRR24166xxx — 50 tumor + 50 normal
```

---

## รายละเอียดแต่ละขั้นตอน

### Step 1 — QC ก่อน Trim
**โฟลเดอร์:** `02_qc_before_trim/`  
ใช้ **FastQC** ตรวจสอบคุณภาพ raw reads เพื่อดูว่ามี adapter contamination หรือ quality ต่ำตรงไหน

```bash
bash 02_qc_before_trim/run_fast.sh
```

---

### Step 2 — Trim Adapter + Low-quality Bases
**โฟลเดอร์:** `03_trim/`  
ใช้ **fastp** ตัด adapter และ bases คุณภาพต่ำออก

```bash
bash 03_trim/trim.sh
```

> ตัวอย่าง: ERR164550 — reads จาก 38.7M → 26.6M หลัง trim

---

### Step 3 — QC หลัง Trim
**โฟลเดอร์:** `04_qc_after_trim/`  
รัน FastQC อีกครั้งหลัง trim เพื่อยืนยันว่าข้อมูลสะอาดแล้ว

```bash
bash 04_qc_after_trim/run_fast.sh
```

---

### Step 4 — Download Genome + Annotation
**โฟลเดอร์:** `01_download_reference/`

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
**โฟลเดอร์:** `05_align/`  
จัดระเบียบ genome ใหม่เพื่อให้ STAR ค้นหาได้เร็ว (ทำครั้งเดียว)

```bash
bash 05_align/build_genome_index.sh
```

---

### Step 6 — Align Reads → Genome
**โฟลเดอร์:** `05_align/`  
ใช้ **STAR** (Spliced Transcripts Alignment to a Reference) จับคู่ reads กลับไปยังตำแหน่งในจีโนม ผลลัพธ์คือไฟล์ BAM

```bash
bash 05_align/Align.sh
```

---

### Step 7 — Count Exons
**โฟลเดอร์:** `06_count_exons/`  
ใช้ **featureCounts** นับจำนวน reads ที่ตกในแต่ละ exon ของแต่ละ sample

```bash
bash 06_count_exons/exon_level_count.sh
```

---

### Step 8 — Merge เป็น Matrix
**โฟลเดอร์:** `07_merge_matrix/`  
รวม count files ของทุก sample เข้าเป็นตาราง (exons × samples)

```bash
# SRR samples เดี่ยว
python3 07_merge_matrix/merge_to_matrix.py

# SRR + ERR รวมกัน (ด้วยวิธี streaming merge ที่ประหยัดแรม)
python3 07_merge_matrix/merge_combined.py
```

---

### Step 8b — Exploratory Data Analysis (EDA)
**โฟลเดอร์:** `08_eda/`  
วิเคราะห์ข้อมูลเชิงสำรวจ เช่น การกระจายตัวของคลาส ขนาดไลบรารี สัดส่วนค่าศูนย์ ความสัมพันธ์ระหว่างค่าเฉลี่ยและความแปรปรวน และการกระจายตัวหลังทำ ComBat batch correction

```bash
python3 08_eda/run_eda.py
```
*อ่านรายงาน EDA ได้ใน [eda_report.md](file:///mnt/c/Users/User/OneDrive/Documents/GitHub/rna_ai_builder/08_eda/eda_report.md)*

---

### Step 9 — Preprocess + Feature Selection
**โฟลเดอร์:** `09_preprocess/`  
คัดเลือก exons เด่นจาก 2 ล้านกว่าตัวให้เหลือ 5,000 ตัวที่มีความแปรปรวนสูงและมีค่า count ที่เชื่อถือได้

```bash
# สำหรับ SRR เดี่ยว
python3 09_preprocess/preprocess.py

# สำหรับ SRR + ERR
python3 09_preprocess/preprocess_combined.py
```

---

### Step 10 — Train ML Model
**โฟลเดอร์:** `10_train/`  
การเปรียบเทียบเชิงวิจัย 5 เฟสเพื่อสร้างโมเดลที่มีประสิทธิภาพสูงสุดและป้องกันปัญหา Data Leakage

```bash
# เทรนเฉพาะ SRR เดี่ยว
python3 10_train/train_model_only_srr.py

# เทรนโมเดลเปรียบเทียบ SRR vs SRR+ERR แบบมี ComBat
python3 10_train/train_model_combined.py

# รัน Cross-validation 80:20 (5-Fold, 10 Repeats) แบบ ComBat Inside Loop
python3 10_train/test_combat_cv8020_real.py
```

#### 📅 เส้นทางการวิจัยและทดลอง (5 เฟส)
1. **Phase 1: Global Setup** - ทำ ComBat ปรับสเกลข้อมูลทั้งหมดพร้อมกัน ได้ผลประเมิน Rule 2 แม่นยำสูงลิ่วถึง **98.04%**
2. **Phase 2: Leakage Fix** - พบ Data Leakage ในการปรับสเกล จึงออกแบบ ComBat-CV แยกฟิตเฉพาะข้อมูลในลูป Train (`error_analysis_complete.py`) ส่งผลให้ความแม่นยำ Rule 2 ตกลงมาเหลือ **86.41%** ในขณะที่โมเดล **SVM (RBF) - 3 Exons** นำด้วยความแม่นยำคงที่ **96.12%**
3. **Phase 3: Y-Scrambling** - ทำการประเมินความนัยสำคัญของผลลัพธ์ (`y_scrambling_leakfree.py`) ได้ Empirical P-value = **0.0196** ยืนยันว่าโมเดลเรียนรู้จากสัญญาณทางชีวภาพจริงที่ปลอดภัยจาก Data Leakage
4. **Phase 4: Raw Evaluation** - รันโมเดลบนข้อมูลดิบที่ไม่ผ่าน Combat เลย (`y_scrambling_nocombat.py`) พบว่า **SVM (RBF) - 3 Exons** ได้ผลลัพธ์เสถียรที่ **96.12%** เช่นเดิม ทำให้ได้โมเดลแนะนำสำหรับใช้งานทางคลินิก (Inference รวดเร็วแบบไม่ต้องมี ComBat)
5. **Phase 5: CV Strategy Evaluation** - เปรียบเทียบประสิทธิภาพแบบ 80:20 CV (5-Fold, 10 Repeats) เทียบกับ LOO-CV (`test_combat_cv8020_real.py`) เพื่อดูความเสถียรและส่วนเบี่ยงเบนมาตรฐาน (Std Dev) ภายใต้ขบวนการ ComBat Inside Loop ปราศจาก Data Leakage

*อ่านบันทึกประวัติการทดลองได้ใน [experimental_timeline.md](file:///mnt/c/Users/User/OneDrive/Documents/GitHub/rna_ai_builder/11_result/experimental_timeline.md)*

---

## ผลลัพธ์
**โฟลเดอร์:** `11_result/`

| ไฟล์ | คืออะไร |
|------|---------|
| [interpretable_if_else_rules.md](file:///mnt/c/Users/User/OneDrive/Documents/GitHub/rna_ai_builder/11_result/interpretable_if_else_rules.md) | รายงานผลลัพธ์กฎ If-Else แบบยีนเด่น |
| [complete_error_analysis_report.md](file:///mnt/c/Users/User/OneDrive/Documents/GitHub/rna_ai_builder/11_result/complete_error_analysis_report.md) | รายงานการวิเคราะห์ตัวอย่างที่ทำนายผิดพลาด |
| [y_scrambling_report_leakfree.md](file:///mnt/c/Users/User/OneDrive/Documents/GitHub/rna_ai_builder/11_result/y_scrambling_report_leakfree.md) | รายงานผลการสลับสลากเดาสุ่มแบบไร้รอยรั่ว |
| [combat_cv_comparison_report.md](file:///mnt/c/Users/User/OneDrive/Documents/GitHub/rna_ai_builder/11_result/combat_cv_comparison_report.md) | รายงานการวิเคราะห์เปรียบเทียบ LOO-CV กับ 80:20 CV แบบ ComBat Inside Loop |
| [experimental_timeline.md](file:///mnt/c/Users/User/OneDrive/Documents/GitHub/rna_ai_builder/11_result/experimental_timeline.md) | บันทึกประวัติและระยะเวลาการทดลองหลัก |
| [pipeline_timeline.md](file:///mnt/c/Users/User/OneDrive/Documents/GitHub/rna_ai_builder/11_result/pipeline_timeline.md) | แผนภาพและสถาปัตยกรรมข้อมูลทางเทคนิคของโมเดล |

และรูปภาพสรุปประสิทธิภาพของโมเดลต่างๆ รวมถึงกราฟ ROC Curves ในแต่ละการทดลอง

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
