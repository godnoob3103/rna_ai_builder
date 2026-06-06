# 🧬 ผังลำดับการประมวลผลข้อมูล (Data Processing Pipeline Timeline)
**โปรเจกต์:** การจำแนกประเภทมะเร็งจากระดับการแสดงออกของ Exon (Exon-level Expression Cancer Classifier)

รายงานฉบับนี้สรุปขั้นตอนการไหลของข้อมูล (Data Flow) และสถาปัตยกรรมของ Pipeline ทั้ง 3 รูปแบบที่เราได้ทำการวิจัยและเปรียบเทียบในโครงการนี้

---

## 📊 ผังภาพรวมการไหลของข้อมูล (End-to-End Pipeline)

```mermaid
graph TD
    A["1. Raw Sequencing Reads (FASTQ)"] -->|Alignment & Exon Quantification| B["2. Exon Counts Matrix (combined_exon_matrix.csv)"]
    B -->|Feature Selection: Top 5,000 exons by variance| C["3. High-Variance Exon Matrix (5,000 Features)"]
    C -->|Log Transformation: log1p(Counts)| D["4. Log-Normalized Data"]
    
    %% Pipeline Choices
    D -->|Pipeline A: Global ComBat| E1["5a. Globally Adjusted Features"]
    D -->|Pipeline B: Fold-wise ComBat| E2["5b. CV Training Fitted ComBat"]
    D -->|Pipeline C: No-ComBat| E3["5c. Raw Log-Normalized Features"]
    
    E1 -->|Static Thresholds| F1["Rule 2 Prediction (98.04% Leaky)"]
    E2 -->|Dynamic Boundary Training| F2["SVM RBF 3-Exon Prediction (96.12% Safe)"]
    E3 -->|Direct Scaled Boundary Training| F3["SVM RBF 3-Exon Prediction (96.12% Safe)"]
```

---

## 🛠️ รายละเอียดขั้นตอนในแต่ละ Pipeline (Step-by-Step Execution)

### 🔴 Pipeline A: Global ComBat Pipeline (โมเดล Leaky ดั้งเดิม)
เป็นสถาปัตยกรรมที่ทำ Batch Correction ก่อนการตัดแบ่งข้อมูลเพื่อประเมินผล ส่งผลให้ประสิทธิภาพที่ได้สูงกว่าความจริง (Overoptimistic)

1. **Alignment:** แปลงข้อมูลดิบ FASTQ ออกมาเป็นตารางนับ Exon Counts `combined_exon_matrix.csv`
2. **Feature Extraction:** คัดกรองเอาเฉพาะยีน Exon 5,000 ตัวแรกที่มีความแปรปรวน (Variance) สูงสุด
3. **Normalization:** ทำการ Normalization ด้วยฟังก์ชันฐานนิยมลอการิทึม $\log(x + 1)$
4. **Global ComBat (จุดรั่วไหล):** นำตัวอย่างทั้งหมด 103 ชิ้นมารัน ComBat Batch Correction พร้อมกัน ข้อมูลของ Test sample จึงรั่วไหลเข้าไปในพารามิเตอร์ของกลุ่ม Train ตั้งแต่จุดนี้
5. **Static Heuristics:** รันกฎตายตัว (Rule 2) หรือเทรนโมเดล ML ทับบนข้อมูลที่ถูกเกลี่ยเรียบร้อยแล้ว
   * **ผลประเมิน:** Rule 2 ได้ **98.04%** (ความแม่นยำสูงเกินจริง)

---

### 🟡 Pipeline B: Leakage-Free ComBat Pipeline (สำหรับการทำ Cross-Validation)
เป็นสถาปัตยกรรมที่ถูกต้องตามหลักการสถิติ สำหรับการทดสอบความแม่นยำเพื่อป้องกันไม่ให้ประเมินประสิทธิภาพโมเดลผิดพลาด

1. **Alignment & Feature Extraction & Log Normalization:** (ทำเหมือน Pipeline A)
2. **Data Splitting (LOO-CV):** แบ่งกลุ่มข้อมูลออกเป็น Training set (102ชิ้น) และ Test sample (1ชิ้น) เสมอในแต่ละลูป
3. **Training ComBat Fit:** สั่งโปรแกรมวิเคราะห์และประมาณค่าสถิติกลุ่ม (Mean/Variance) ของ ComBat **เฉพาะจากข้อมูล 102 ชิ้นใน Training set เท่านั้น**
4. **Transform:** 
   * นำพารามิเตอร์ ComBat จากกลุ่ม Train ไปปรับแก้สเกลของ Training set เอง
   * นำพารามิเตอร์เดิมนั้นไปปรับแก้ให้กับ Test sample 1 ชิ้นที่อยู่ภายนอกแบบเดี่ยวๆ (เพื่อรักษาความเป็นกลางเสมือนตัวอย่างที่มาจากอนาคต)
5. **Standard Scaling:** เทรน StandardScaler บนข้อมูล Train แล้วนำมา Transform ตัว Test
6. **Model Training & Prediction:** เทรน SVM (RBF) บนยีนเด่น 3 ตัว และทำนายผลตัวอย่างทดสอบ
   * **ผลประเมิน:** SVM 3-Exon ได้ **96.12%** (ผ่าน Y-Scrambling ตรวจสอบแล้วว่าปลอดภัย)

---

### 🟢 Pipeline C: Raw No-ComBat Pipeline (สำหรับใช้จริงในคลินิก - Deployed Model)
เป็นสถาปัตยกรรมสุดท้ายที่แนะนำให้ใช้งานจริง เนื่องจากให้ผลลัพธ์สูงเท่ากันกับแบบแก้ไข ComBat แต่ตัดกระบวนการจัดตำแหน่งสเกลที่ยุ่งยากออกไปจนหมด

1. **Alignment:** นำเข้าตัวอย่างใหม่เดี่ยวๆ (New sample) และดึงข้อมูลจำนวนนับของยีน 3 ตัวหลักออกมา:
   * `ENSG00000164266_620977`
   * `ENSG00000019169_316124`
   * `ENSG00000204305_741180`
2. **Log Normalization:** คำนวณค่า $\log(x + 1)$ บนระดับการแสดงออกของ 3 ยีนนี้
3. **Standard Scaling (Inference):** ปรับจูนระดับสเกล (Z-score) โดยใช้พารามิเตอร์เฉลี่ย ($\mu, \sigma$) ที่จำลองขึ้นจาก Training cohort เดิม
4. **Dynamic Classifier Prediction:** ป้อนค่ายีนที่แปลงแล้วเข้าสู่โมเดล **SVM (RBF) - 3 Exons** ที่เซฟเอาไว้ เพื่อทำนายผลมะเร็ง/ปกติทันที
   * **ผลประเมิน:** ได้ความแม่นยำ **96.12%** (แม่นยำสูงและมีความเสถียรที่สุด ไม่ขึ้นกับ batch correction ใดๆ)
